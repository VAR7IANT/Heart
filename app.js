/*
 * Particle heart — WebGL2 renderer.
 *
 * Pipeline
 *   1. A static particle buffer is generated once on the CPU: points on (and
 *      inside) a smooth implicit 3D heart surface, plus emitted aura and a
 *      sparse field of distant embers.
 *   2. All motion happens in the vertex shader: surface particles stream
 *      along the skin following a divergence-free flow field and are
 *      re-projected onto the surface analytically, aura particles are
 *      advected outward, and every particle feels the lub-dub pressure wave.
 *   3. Particles are blended additively into a half-float HDR target, so
 *      overlap never clips or flickers. Rim lighting (Fresnel) keeps the
 *      interior dark and the silhouette luminous.
 *   4. A dual-filter bloom chain and an ACES tone map / vignette / dither
 *      composite produce the final image.
 */
(() => {
  'use strict';

  const { HeartRhythm, HeartbeatAudio, beatShape } = window.Heartbeat;
  const canvas = document.getElementById('heart');
  const root = document.documentElement;
  const motionQuery = matchMedia('(prefers-reduced-motion: reduce)');
  const rhythm = new HeartRhythm();
  const audio = new HeartbeatAudio();

  const glOptions = {
    alpha: false, antialias: false, depth: false, stencil: false,
    premultipliedAlpha: false, preserveDrawingBuffer: false,
    powerPreference: 'high-performance'
  };
  const gl = canvas.getContext('webgl2', glOptions);
  if (!gl) return startFallback();

  /* ------------------------------------------------------------------ *
   * Heart geometry                                                     *
   * ------------------------------------------------------------------ */

  // Taubin's heart surface (x² + 9/4·y² + z² − 1)³ = z³·(x² + 9/80·y²),
  // remapped so that model Y is up and model Z is depth. Taking the real
  // cube root of both sides gives the same surface as
  //     F = (x² + 9/4·y² + z² − 1) − z·∛(x² + 9/80·y²) = 0,
  // whose gradient — unlike the classic sextic's — never vanishes along the
  // heart's equator. That degeneracy is what produces the bright horizontal
  // "seam" in naive renderings; with F, distances and normals are smooth.
  const DEPTH = 1.3;   // fuller than the classic surface: rounder in profile
  const LIFT = 0.118;  // centers the shape vertically

  function field(X, Y, Z, out) {
    const x = X, y = Z / DEPTH, z = Y + LIFT;
    const s = x * x + 0.1125 * y * y;
    const k = Math.cbrt(s), q = z / (3 * k * k + 1e-5);
    out[0] = 2 * x * (1 - q);
    out[1] = 2 * z - k;
    out[2] = y * (4.5 - 0.225 * q) / DEPTH;
    return x * x + 2.25 * y * y + z * z - 1 - z * k;
  }

  function mulberry32(seed) {
    return () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const BOX = [1.2, 1.18, 0.92];

  /** Uniform points on the surface: uniform samples in a thin band are kept
   *  only if their true distance to the surface (measured after Newton
   *  projection) is inside the band, so the area density stays even all
   *  the way into the cleft and the tip. */
  function sampleSurface(count, rand) {
    const grad = [0, 0, 0], normal = [0, 0, 0];
    const out = new Float32Array(count * 6);
    const band = 0.02;
    let n = 0;
    while (n < count) {
      const x = (rand() * 2 - 1) * BOX[0];
      const y = (rand() * 2 - 1) * BOX[1];
      const z = (rand() * 2 - 1) * BOX[2];
      let f = field(x, y, z, grad);
      let gg = grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2];
      if (Math.abs(f) > band * 1.05 * Math.sqrt(gg)) continue;
      let px = x, py = y, pz = z, converged = false;
      for (let i = 0; i < 24 && gg > 1e-12; i++) {
        const s = f / gg, sx = grad[0] * s, sy = grad[1] * s, sz = grad[2] * s;
        px -= sx; py -= sy; pz -= sz;
        if (sx * sx + sy * sy + sz * sz < 1e-12) { converged = true; break; }
        f = field(px, py, pz, grad);
        gg = grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2];
      }
      if (!converged || Math.hypot(px - x, py - y, pz - z) > band) continue;
      field(px, py, pz, grad);
      const inv = 1 / Math.hypot(grad[0], grad[1], grad[2]);
      normal[0] = grad[0] * inv; normal[1] = grad[1] * inv; normal[2] = grad[2] * inv;
      const o = n++ * 6;
      out[o] = px; out[o + 1] = py; out[o + 2] = pz;
      out[o + 3] = normal[0]; out[o + 4] = normal[1]; out[o + 5] = normal[2];
    }
    return out;
  }

  // Per particle: position (3), normal (3), data (4: seed, kind, a, b).
  const STRIDE = 10;
  const KIND = { surface: 0, interior: 1, aura: 2, ember: 3, glint: 4, comet: 5, shock: 6, spark: 7, disc: 8, bokeh: 9 };

  /**
   * Builds the base layers (shuffled, so any prefix is a uniform subset for
   * adaptive quality) followed by the effect layers (sorted by kind, drawn
   * in full as a second range so each GPU batch runs a single code path).
   */
  function buildParticles(total, fx) {
    const rand = mulberry32(0x5eed1e);
    const counts = {
      surface: Math.round(total * 0.6),
      interior: Math.round(total * 0.15),
      aura: Math.round(total * 0.205)
    };
    counts.ember = total - counts.surface - counts.interior - counts.aura;
    const surface = sampleSurface(counts.surface, rand);
    const fxTotal = fx.glint + fx.comet + fx.shock + fx.spark + fx.disc + fx.bokeh;
    const data = new Float32Array((total + fxTotal) * STRIDE);
    let n = 0;
    const push = (x, y, z, nx, ny, nz, kind, a, b) => {
      const o = n++ * STRIDE;
      data[o] = x; data[o + 1] = y; data[o + 2] = z;
      data[o + 3] = nx; data[o + 4] = ny; data[o + 5] = nz;
      data[o + 6] = rand(); data[o + 7] = kind; data[o + 8] = a; data[o + 9] = b;
    };

    for (let i = 0; i < counts.surface; i++) {
      const o = i * 6;
      push(surface[o], surface[o + 1], surface[o + 2], surface[o + 3], surface[o + 4], surface[o + 5],
        KIND.surface, rand(), rand());
    }

    // Interior: denser just beneath the skin, sparse at the core.
    const grad = [0, 0, 0];
    for (let i = 0; i < counts.interior;) {
      const x = (rand() * 2 - 1) * BOX[0], y = (rand() * 2 - 1) * BOX[1], z = (rand() * 2 - 1) * BOX[2];
      const f = field(x, y, z, grad);
      if (f >= 0) continue;
      const len = Math.hypot(grad[0], grad[1], grad[2]) + 1e-6;
      const depth = Math.min(1, -f / len / 0.35);
      if (rand() > 0.25 + 0.75 * Math.exp(-depth * 3)) continue;
      push(x, y, z, grad[0] / len, grad[1] / len, grad[2] / len, KIND.interior, rand(), depth);
      i++;
    }

    // Aura: emitted from random surface points.
    for (let i = 0; i < counts.aura; i++) {
      const o = Math.floor(rand() * counts.surface) * 6;
      push(surface[o], surface[o + 1], surface[o + 2], surface[o + 3], surface[o + 4], surface[o + 5],
        KIND.aura, rand(), rand());
    }

    // Embers: drifting dust that fills the whole view in depth. Positions
    // are stored in normalized screen space (x, y in -1..1) plus depth; the
    // shader maps them through the live camera frustum, so the dust fills
    // portrait phones and wide monitors alike.
    for (let i = 0; i < counts.ember; i++) {
      push(rand() * 2 - 1, rand() * 2 - 1, -7 + rand() * 9.5, 0, 1, 0, KIND.ember, rand(), rand());
    }

    // Global shuffle (so reduced draw counts keep every layer's ratio).
    const swap = new Float32Array(STRIDE);
    for (let i = total - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      swap.set(data.subarray(i * STRIDE, i * STRIDE + STRIDE));
      data.copyWithin(i * STRIDE, j * STRIDE, j * STRIDE + STRIDE);
      data.set(swap, j * STRIDE);
    }

    const fromSurface = (kind, upperBias) => {
      let o;
      do o = Math.floor(rand() * counts.surface) * 6;
      while (upperBias && surface[o + 1] < -0.2 && rand() < upperBias);
      push(surface[o], surface[o + 1], surface[o + 2], surface[o + 3], surface[o + 4], surface[o + 5],
        kind, rand(), rand());
    };

    // Glints: star-shaped flares that bloom briefly on the skin.
    for (let i = 0; i < fx.glint; i++) fromSurface(KIND.glint, 0);

    // Comets: 3 tilted orbits × 2 comets, each a bright head with a tail
    // that widens and fades (head-biased sampling keeps heads dense).
    // position = (ring, comet phase, tail u); normal = jitter direction.
    const perComet = Math.floor(fx.comet / 6);
    for (let ring = 0; ring < 3; ring++) {
      for (let c = 0; c < 2; c++) {
        for (let i = 0; i < perComet; i++) {
          const u = Math.pow(rand(), 1.7);
          const jx = rand() * 2 - 1, jy = rand() * 2 - 1, jz = rand() * 2 - 1;
          push(ring, c * 0.5 + ring * 0.17, u, jx, jy, jz, KIND.comet, rand(), rand());
        }
      }
    }
    for (let i = n; i < total + fx.glint + fx.comet; i++) push(0, 0, 1, 0, 0, 0, KIND.comet, 0, 0);

    // Shockwave: a copy of the skin thrown outward on each lub and dub
    // (the shader lights only its rim, so it reads as an expanding outline).
    for (let i = 0; i < fx.shock; i++) fromSurface(KIND.shock, 0);

    // Sparks: bright embers flung upward off the turning heart.
    for (let i = 0; i < fx.spark; i++) fromSurface(KIND.spark, 0.7);

    // Disc: a wide, two-armed spiral of light the heart floats inside.
    // position = (radius, angle, height).
    for (let i = 0; i < fx.disc; i++) {
      const r = 1.85 + Math.pow(rand(), 1.25) * 3.1;
      let theta = rand() * Math.PI * 2;
      if (rand() < 0.62) {
        const g = (rand() + rand() + rand() - 1.5) * 0.55;
        theta = (i & 1) * Math.PI + Math.log(r) * 2.6 + g;
      }
      const h = (rand() + rand() - 1) * 0.05 * r;
      push(r, theta, h, 0, 0, 0, KIND.disc, rand(), rand());
    }

    // Bokeh: a few soft, out-of-focus hearts drifting up in the distance.
    // position = (normalized screen x, depth).
    for (let i = 0; i < fx.bokeh;) {
      const z = -4.5 + rand() * 6;
      const x = rand() * 2 - 1;
      if (z > -2.2 && Math.abs(x) < 0.4) continue;
      push(x, z, 0, 0, 0, 0, KIND.bokeh, rand(), rand());
      i++;
    }

    return data;
  }

  /* ------------------------------------------------------------------ *
   * Shaders                                                            *
   * ------------------------------------------------------------------ */

  const COMMON = `
    float pulse(float x, float tau) {
      x = max(x, 0.0);
      float u = x / (2.0 * tau);
      return u * u * exp(2.0 - x / tau);
    }
    float beatAt(float age) { return pulse(age, 0.045) + 0.56 * pulse(age - 0.30, 0.04); }
  `;

  const PARTICLE_VS = `#version 300 es
    precision highp float;
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aNormal;
    layout(location = 2) in vec4 aData;   // seed, kind, a, b

    uniform mat4 uViewProj;
    uniform mat3 uModel;
    uniform vec3 uOffset;
    uniform vec3 uCamera;
    uniform float uTime;
    uniform vec2 uBeatAge;
    uniform float uIntro;
    uniform float uFocal;
    uniform float uFocus;
    uniform vec2 uFrustum;    // tan(fov/2) * (aspect, 1)
    uniform float uDiscGain;  // compensates the smaller disc on phones
    uniform float uMaxSize;
    uniform float uSize;
    uniform float uGain;
    uniform vec4 uRay;        // pointer ray direction (xyz), presence (w)
    uniform vec4 uRipple;     // ripple ray direction (xyz), age (w)

    out vec3 vColor;
    out float vSoft;
    out float vStar;

    ${COMMON}
    float beat(float delay) { return beatAt(uBeatAge.x - delay) + beatAt(uBeatAge.y - delay); }

    // Divergence-free (ABC-style) flow: each component ignores its own axis.
    vec3 flow(vec3 p, float t) {
      vec3 q = p * 1.6 + vec3(t * 0.11, t * 0.07, -t * 0.09);
      vec3 v = vec3(sin(q.z) + cos(q.y), sin(q.x) + cos(q.z), sin(q.y) + cos(q.x));
      vec3 r = p * 3.7 + vec3(-t * 0.17, t * 0.13, t * 0.05);
      v += 0.45 * vec3(sin(r.y + 1.3) + cos(r.z), sin(r.z + 2.1) + cos(r.x), sin(r.x + 0.7) + cos(r.y));
      return v;
    }

    const float DEPTH = ${DEPTH.toFixed(4)};
    const float LIFT = ${LIFT.toFixed(4)};
    vec4 heart(vec3 P) {
      float x = P.x, y = P.z / DEPTH, z = P.y + LIFT;
      float k = pow(x * x + 0.1125 * y * y, 1.0 / 3.0);
      float q = z / (3.0 * k * k + 1e-5);
      return vec4(2.0 * x * (1.0 - q), 2.0 * z - k, y * (4.5 - 0.225 * q) / DEPTH,
                  x * x + 2.25 * y * y + z * z - 1.0 - z * k);
    }

    vec3 hash3(float s) {
      return fract(sin(vec3(s * 127.1, s * 311.7, s * 74.7)) * 43758.5453) * 2.0 - 1.0;
    }

    mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }

    // The heartbeat deformation shared by everything attached to the heart:
    // swell, a slight settle and a gentle wringing twist.
    vec3 deform(vec3 p, float b) {
      p *= 1.0 + 0.068 * b + 0.006 * sin(uTime * 0.55);
      p.y -= 0.012 * b;
      p.xz = rot(0.05 * b * p.y) * p.xz;
      return p;
    }

    // Projection, point size, depth of field and culling for every particle.
    void emit(vec3 world, vec3 col, float size, float sharp, float star) {
      gl_Position = uViewProj * vec4(world, 1.0);
      float w = max(gl_Position.w, 0.05);
      float px = size * uSize * uFocal / w;
      float coc = star > 0.5 && star < 1.5 ? 0.0 : abs(w - uFocus) * uFocal * 0.0024;
      float blurred = sqrt(px * px + coc * coc);
      float energyScale = (px * px) / (blurred * blurred);
      px = blurred;
      // Sub-pixel points are widened instead of shrunk: no shimmer.
      float minPx = 1.4;
      if (px < minPx) { energyScale *= (px * px) / (minPx * minPx); px = minPx; }
      if (px > uMaxSize) { energyScale *= (px * px) / (uMaxSize * uMaxSize); px = uMaxSize; }
      vColor = col * energyScale * uGain;
      vSoft = mix(sharp, 2.2, clamp(coc / (coc + px * 0.5 + 1.0), 0.0, 1.0));
      vStar = star;
      gl_PointSize = px;
      // Fully faded particles are moved outside the clip volume so they
      // cost no fill rate at all.
      if (max(vColor.r, max(vColor.g, vColor.b)) < 1e-4) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        gl_PointSize = 0.0;
      }
    }

    // Effect layers: glints, comets, beat shockwave and rising sparks.
    void effect(float seed, float kind, float ra, float rb) {
      float t = uTime;
      float fade = smoothstep(0.55, 1.0, uIntro);
      float b = beat(0.04);
      vec3 V;
      if (kind < 4.5) {
        // Glint: a lens-like star that blooms on the skin and fades again.
        float life = 3.0 + 4.0 * ra;
        float age = fract(t / life + seed);
        float env = pow(sin(3.14159 * age), 28.0);
        vec3 world = uModel * (deform(aPos, b) + aNormal * 0.012) + uOffset;
        vec3 N = normalize(uModel * aNormal);
        V = normalize(uCamera - world);
        float ndv = dot(N, V);
        float e = env * smoothstep(0.0, 0.35, ndv) * (0.35 + 1.65 * (1.0 - abs(ndv)))
                * (1.0 + 1.0 * b) * fade;
        vec3 col = mix(vec3(1.0, 0.06, 0.09), vec3(1.0, 0.36, 0.34), env) * e * 0.4;
        emit(world, col, (13.0 + 12.0 * rb) * (0.55 + 0.45 * env), 4.0, 1.0);
      } else if (kind < 5.5) {
        // Comet: a bright head and widening tail on a tilted orbit.
        float ring = aPos.x, u = aPos.z;
        float dir = ring == 1.0 ? -1.0 : 1.0;
        float radius = 1.5 + 0.17 * ring + 0.035 * b;
        float head = aPos.y * 6.2831853 + t * dir * (0.6 - 0.09 * ring);
        float theta = head - dir * u * 2.3;
        vec3 q = vec3(cos(theta), 0.0, sin(theta)) * radius;
        q.xy = rot((ring - 1.0) * 0.62 + 0.1) * q.xy;
        q.yz = rot(0.3 + 0.1 * ring) * q.yz;
        q += aNormal * (0.01 + 0.085 * u * u) + flow(q, t * 0.3) * 0.04 * u;
        vec3 world = q + uOffset;
        // Dimmed while it passes behind the heart.
        float behind = smoothstep(0.1, -0.4, world.z)
                     * smoothstep(1.3, 0.75, length(world.xy * vec2(0.95, 1.0)));
        float e = (pow(1.0 - u, 2.6) * 0.75 + exp(-u * 70.0) * 1.5)
                * (1.0 - 0.9 * behind) * (1.0 + 0.6 * b) * fade;
        vec3 col = mix(vec3(1.0, 0.03, 0.055), vec3(1.0, 0.42, 0.36), exp(-u * 22.0)) * e * 0.5;
        emit(world, col, (0.9 + 0.7 * ra) * (1.3 - 0.6 * u), 3.6, 0.0);
      } else if (kind < 6.5) {
        // Shockwave: the heart's own silhouette thrown outward on each
        // lub (and, softer, each dub); only the rim of the shell is lit.
        float dub = step(rb, 0.33);
        float age = uBeatAge.x - dub * 0.30;
        float life = mix(0.88, 0.6, dub);
        float k = clamp(age / life, 0.0, 1.0);
        float grow = 1.0 - pow(1.0 - k, 3.0);
        vec3 q = aPos * (1.05 + mix(0.6, 0.36, dub) * grow) + aNormal * 0.02
               + hash3(seed) * (0.006 + 0.05 * k);
        vec3 world = uModel * q + uOffset;
        V = normalize(uCamera - world);
        float rim = pow(1.0 - abs(dot(normalize(uModel * aNormal), V)), 3.0);
        float e = step(0.0, age) * step(age, life) * smoothstep(0.0, 0.06, age)
                * pow(1.0 - k, 2.0) * mix(1.0, 0.55, dub) * rim * fade * 1.5;
        vec3 col = mix(vec3(1.0, 0.28, 0.22), vec3(1.0, 0.025, 0.05), k) * e;
        emit(world, col, (0.9 + 0.6 * ra) * (1.0 + 0.8 * k), 3.5, 0.0);
      } else if (kind < 7.5) {
        // Spark: a hot ember flung off the turning heart, rising and cooling.
        float life = 2.6 + 2.8 * ra;
        float age = fract(t / life + seed);
        vec3 world = uModel * (deform(aPos, b) + aNormal * (0.04 + 0.28 * age)) + uOffset;
        world.y += age * age * (0.8 + 0.8 * rb);
        world += flow(world * 0.7, t * 0.4) * 0.1 * age;
        float tw = 0.75 + 0.25 * sin(t * 2.1 + seed * 50.0);
        float e = smoothstep(0.0, 0.08, age) * pow(1.0 - age, 1.6) * tw * fade * 1.4;
        vec3 col = mix(vec3(1.0, 0.3, 0.2), vec3(1.0, 0.035, 0.06), age) * e;
        emit(world, col, 0.75 + 0.55 * rb, 4.5, 0.0);
      } else if (kind < 8.5) {
        // Disc: rigid spiral pattern turning slowly (arms never smear),
        // tilted so its near side sweeps beneath the heart; each beat sends
        // a ring of light outward across it.
        float r = aPos.x;
        float theta = aPos.y + t * 0.1 + 0.05 * sin(t * 0.27 + seed * 20.0);
        vec3 q = vec3(cos(theta) * r, aPos.z, sin(theta) * r);
        q.yz = rot(-0.38) * q.yz;
        q.y -= 0.3;
        q += flow(q * 0.45, t * 0.15) * 0.07;
        vec3 world = q + uOffset;
        float behind = smoothstep(0.1, -0.4, world.z)
                     * smoothstep(1.3, 0.75, length(world.xy * vec2(0.95, 1.0)));
        float wave = beat(r * 0.13);
        float shape = smoothstep(1.85, 2.4, r) * (1.0 - smoothstep(3.7, 4.95, r));
        float e = (0.1 + 0.16 * rb * rb + 0.32 * wave) * (ra > 0.97 ? 3.5 : 1.0) * uDiscGain
                * shape * (1.0 - 0.85 * behind) * fade;
        vec3 col = mix(vec3(0.55, 0.015, 0.04), vec3(1.0, 0.07, 0.09), rb * 0.6 + wave * 0.4) * e;
        emit(world, col, 0.8 + 0.8 * ra, 3.4, 0.0);
      } else {
        // Bokeh hearts: slow, swaying, deeply out of focus.
        float life = 18.0 + 12.0 * ra;
        float age = fract(t / life + seed);
        vec2 span = uFrustum * (uFocus - aPos.y);
        vec3 world = vec3(aPos.x * span.x + sin(t * 0.35 + seed * 30.0) * 0.3,
                          mix(-1.2, 1.2, age) * span.y, aPos.y);
        world += uOffset;
        float e = sin(3.14159 * age) * (0.05 + 0.07 * rb) * (1.0 + 0.4 * b) * fade;
        emit(world, vec3(1.0, 0.05, 0.08) * e * 0.8, 17.0 + 15.0 * rb, 2.0, 2.0);
      }
    }


    void main() {
      float seed = aData.x, kind = aData.y, ra = aData.z, rb = aData.w;
      if (kind > 3.5) { effect(seed, kind, ra, rb); return; }
      vec3 p = aPos, n = aNormal;
      float alpha = 1.0, size = 1.0, sharp = 4.0;
      float t = uTime;
      vec3 col;
      float wave = 0.0;
      float coreDist = length(aPos * vec3(1.0, 1.0, 1.3));

      if (kind < 0.5) {
        // Surface: stream along the skin, then snap back onto it.
        float life = 6.0 + 6.0 * ra;
        float age = fract(t / life + seed);
        float travel = age * (0.08 + 0.1 * rb);
        for (int i = 0; i < 3; i++) {
          vec3 v = flow(p, t * 0.25);
          v -= n * dot(v, n);
          p += v * (travel / 3.0) * 0.5;
          vec4 h = heart(p);
          float gg = dot(h.xyz, h.xyz);
          // Newton step back onto the surface.
          if (gg > 1e-6) { p -= h.xyz * (h.w / gg); n = h.xyz * inversesqrt(gg); }
        }
        p -= n * rb * rb * 0.045;
        alpha = smoothstep(0.0, 0.18, age) * smoothstep(1.0, 0.72, age);
        size = 0.75 + 0.6 * ra;
      } else if (kind < 1.5) {
        // Interior: slow, deep drift.
        p += flow(p * 0.8, t * 0.2) * 0.022;
        size = 0.9 + 0.8 * ra;
        sharp = 3.0;
      } else if (kind < 2.5) {
        // Aura: lifted off the surface and carried outward by the flow.
        float life = 2.8 + 2.6 * ra;
        float age = fract(t / life + seed);
        float dist = age * (0.22 + 0.3 * rb);
        for (int i = 0; i < 4; i++) {
          vec3 v = n * 1.1 + flow(p * 0.9, t * 0.3) * 0.5 + vec3(0.0, 0.12, 0.0);
          p += v * (dist / 4.0);
        }
        alpha = smoothstep(0.0, 0.1, age) * pow(1.0 - age, 2.2);
        size = 0.85 + 0.9 * ra + age * 0.8;
        sharp = 3.2;
        wave = age;
      } else {
        // Embers: distant, slow and softly out of focus.
        float life = 14.0 + 10.0 * ra;
        float age = fract(t / life + seed);
        p = vec3(aPos.xy * uFrustum * (uFocus - aPos.z) * 1.08, aPos.z);
        p += flow(p * 0.3, t * 0.05) * 0.22 + vec3(0.0, (age - 0.5) * 0.9, 0.0);
        // Keep the space immediately around the heart clear.
        alpha *= smoothstep(1.5, 2.1, length(vec3(p.xy, p.z * 1.4)));
        alpha *= sin(3.14159 * age) * (rb > 0.9 ? 3.0 : 1.0);
        size = 0.9 + 1.4 * rb;
        sharp = 2.6;
      }

      // Lub-dub: a pressure wave from the core outward (core moves first).
      float b = beat(coreDist * 0.06 + (kind > 1.5 && kind < 2.5 ? wave * 0.25 : 0.0));
      if (kind < 2.5) {
        p = deform(p, b);
        if (kind > 1.5) p += n * b * 0.07 * (1.0 - wave);
      }

      // Intro: particles gather from a wide, swirling cloud.
      float k = clamp(uIntro * 1.55 - seed * 0.55, 0.0, 1.0);
      k = k * k * (3.0 - 2.0 * k);
      if (k < 1.0) {
        vec3 far = aPos * (2.4 + 1.8 * ra) + hash3(seed) * 1.4 + flow(aPos, seed * 10.0) * 0.4;
        if (kind < 2.5) p = mix(far, p, k);
        alpha *= k;
      }

      // Embers stay fixed in the world so the backdrop is calm while the
      // heart turns; everything else rotates with the heart.
      vec3 world = (kind > 2.5 ? p : uModel * p) + uOffset;

      // Pointer: particles near the pointer ray part gently, and a tap sends
      // a soft ring outward around it.
      vec3 rel = world - uCamera;
      float glow = 0.0;
      if (uRay.w > 0.001) {
        vec3 perp = rel - uRay.xyz * dot(rel, uRay.xyz);
        float d2 = dot(perp, perp);
        float f = exp(-d2 / 0.07) * uRay.w;
        world += perp * inversesqrt(d2 + 1e-4) * f * 0.05;
        glow += f * 0.45;
      }
      if (uRipple.w < 2.5) {
        vec3 perp = rel - uRipple.xyz * dot(rel, uRipple.xyz);
        float d = length(perp);
        float ring = d - uRipple.w * 1.5;
        float f = exp(-ring * ring / 0.02) * exp(-uRipple.w * 1.6);
        world += perp / (d + 1e-3) * f * 0.06;
        glow += f * 0.9;
      }

      // Shading.
      vec3 N = normalize(uModel * n);
      vec3 V = normalize(uCamera - world);
      float ndv = dot(N, V);
      float fres = 1.0 - abs(ndv);
      vec3 deep = vec3(0.24, 0.006, 0.018);
      vec3 crimson = vec3(1.0, 0.028, 0.045);
      vec3 hot = vec3(1.0, 0.22, 0.12);
      float energy;
      if (kind < 0.5) {
        float rim = pow(fres, 2.6);
        float facing = mix(0.42, 1.0, smoothstep(-0.35, 0.35, ndv));
        vec3 L = normalize(vec3(-0.55, 0.65, 0.55));
        float spec = pow(max(dot(reflect(-L, N), V), 0.0), 5.0) * smoothstep(0.0, 0.3, ndv);
        float key = max(dot(N, L), 0.0);
        energy = (0.02 + key * key * 0.05 + rim * (2.1 + 1.6 * b) + spec * 0.16) * facing;
        col = mix(deep, crimson, smoothstep(0.0, 0.55, rim + spec)) * energy
            + hot * pow(rim, 5.0) * (0.35 + 1.1 * b) * facing;
      } else if (kind < 1.5) {
        energy = (0.034 + 0.04 * b) * (1.0 - 0.6 * rb);
        col = vec3(0.4, 0.012, 0.03) * energy;
      } else if (kind < 2.5) {
        energy = (0.3 + 0.5 * b) * (0.12 + 0.88 * pow(fres, 2.0));
        col = mix(crimson, deep, wave) * energy;
      } else {
        col = vec3(0.8, 0.05, 0.08) * 0.2;
      }
      emit(world, col * alpha * (1.0 + glow), size, sharp, 0.0);
    }
  `;

  const PARTICLE_FS = `#version 300 es
    precision mediump float;
    in vec3 vColor;
    in float vSoft;
    in float vStar;
    out vec4 outColor;
    void main() {
      vec2 c = gl_PointCoord * 2.0 - 1.0;
      float r2 = dot(c, c);
      float e = exp(-vSoft);
      float a = max((exp(-r2 * vSoft) - e) / (1.0 - e), 0.0);
      if (vStar > 1.5) {
        // Soft heart bokeh: a smooth heart SDF with a faint brighter rim.
        vec2 p = vec2(abs(c.x), 0.62 - c.y) * 0.95;
        float d;
        if (p.y + p.x > 1.0) d = length(p - vec2(0.25, 0.75)) - 0.35355;
        else d = sqrt(min(dot(p - vec2(0.0, 1.0), p - vec2(0.0, 1.0)),
                          dot(p - 0.5 * max(p.x + p.y, 0.0), p - 0.5 * max(p.x + p.y, 0.0))))
                 * sign(p.x - p.y);
        float body = smoothstep(0.1, -0.16, d);
        a = body * (0.7 + 0.45 * smoothstep(-0.25, -0.03, d));
      } else if (vStar > 0.5) {
        // Four-point star with fainter diagonals, tapering to the edge.
        vec2 ac = abs(c);
        vec2 d = abs(vec2(c.x + c.y, c.x - c.y)) * 0.7071;
        float rays = exp(-ac.y * 16.0) * (1.0 - ac.x) + exp(-ac.x * 16.0) * (1.0 - ac.y);
        rays += 0.35 * (exp(-d.y * 22.0) * max(1.0 - d.x * 1.7, 0.0)
                      + exp(-d.x * 22.0) * max(1.0 - d.y * 1.7, 0.0));
        a = exp(-r2 * 26.0) * 1.6 + exp(-r2 * 6.0) * 0.25 + rays * 0.5 * max(1.0 - r2, 0.0);
      }
      outColor = vec4(vColor * a, 1.0);
    }
  `;

  const QUAD_VS = `#version 300 es
    precision highp float;
    out vec2 vUv;
    void main() {
      vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
      vUv = p;
      gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    }
  `;

  // Dual-filter (Kawase) bloom: cheap, wide and stable — ideal on mobile.
  const DOWN_FS = `#version 300 es
    precision mediump float;
    in vec2 vUv;
    uniform sampler2D uTex;
    uniform vec2 uTexel;
    uniform float uPrefilter;
    out vec4 outColor;
    vec3 prefilter(vec3 c) {
      float l = max(c.r, max(c.g, c.b));
      float knee = 0.3, threshold = 0.32;
      float soft = clamp(l - threshold + knee, 0.0, 2.0 * knee);
      soft = soft * soft / (4.0 * knee + 1e-4);
      return c * max(soft, l - threshold) / max(l, 1e-4);
    }
    void main() {
      vec2 o = uTexel * 0.5;
      vec3 c = texture(uTex, vUv).rgb * 4.0;
      c += texture(uTex, vUv + vec2(-o.x, -o.y)).rgb;
      c += texture(uTex, vUv + vec2( o.x, -o.y)).rgb;
      c += texture(uTex, vUv + vec2(-o.x,  o.y)).rgb;
      c += texture(uTex, vUv + vec2( o.x,  o.y)).rgb;
      c *= 0.125;
      if (uPrefilter > 0.5) c = prefilter(c);
      outColor = vec4(c, 1.0);
    }
  `;

  const UP_FS = `#version 300 es
    precision mediump float;
    in vec2 vUv;
    uniform sampler2D uTex;
    uniform vec2 uTexel;
    uniform float uWeight;
    out vec4 outColor;
    void main() {
      vec2 o = uTexel * 0.5;
      vec3 c = texture(uTex, vUv + vec2(-o.x * 2.0, 0.0)).rgb;
      c += texture(uTex, vUv + vec2(-o.x, o.y)).rgb * 2.0;
      c += texture(uTex, vUv + vec2(0.0, o.y * 2.0)).rgb;
      c += texture(uTex, vUv + vec2(o.x, o.y)).rgb * 2.0;
      c += texture(uTex, vUv + vec2(o.x * 2.0, 0.0)).rgb;
      c += texture(uTex, vUv + vec2(o.x, -o.y)).rgb * 2.0;
      c += texture(uTex, vUv + vec2(0.0, -o.y * 2.0)).rgb;
      c += texture(uTex, vUv + vec2(-o.x, -o.y)).rgb * 2.0;
      outColor = vec4(c * (uWeight / 12.0), 1.0);
    }
  `;

  // Anamorphic streak at quarter resolution: only the brightest highlights
  // (glints, comet heads, the hottest rim) smear into a horizontal flare.
  const STREAK_FS = `#version 300 es
    precision mediump float;
    in vec2 vUv;
    uniform sampler2D uTex;
    uniform float uStep;
    out vec4 outColor;
    void main() {
      vec3 c = vec3(0.0);
      for (int i = -10; i <= 10; i++) {
        float fi = float(i);
        vec3 s = texture(uTex, vUv + vec2(fi * uStep, 0.0)).rgb;
        c += max(s - 0.05, 0.0) * exp(-abs(fi) * 0.24);
      }
      outColor = vec4(c, 1.0);
    }
  `;

  // Light rays: a radial blur of the bloom toward the heart's center, so the
  // glow streams outward into the surrounding darkness in soft shafts.
  const RAYS_FS = `#version 300 es
    precision highp float;
    in vec2 vUv;
    uniform sampler2D uTex;
    uniform vec2 uCenter;
    out vec4 outColor;
    void main() {
      vec2 delta = (uCenter - vUv) * (0.6 / 32.0);
      // Per-pixel jittered start hides the discrete sample steps (which
      // otherwise show up as beaded copies of bright points).
      float j = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
      vec2 uv = vUv + delta * j;
      float w = 1.0;
      vec3 c = vec3(0.0);
      for (int i = 0; i < 32; i++) {
        c += max(texture(uTex, uv).rgb - 0.06, 0.0) * w;
        uv += delta;
        w *= 0.95;
      }
      outColor = vec4(c / 16.0, 1.0);
    }
  `;

  const COMPOSITE_FS = `#version 300 es
    precision highp float;
    in vec2 vUv;
    uniform sampler2D uScene;
    uniform sampler2D uBloom;
    uniform sampler2D uStreak;
    uniform sampler2D uRays;
    uniform float uRayStrength;
    uniform vec2 uAspect;
    uniform vec2 uCenter;
    uniform float uBeat;
    uniform float uBloomStrength;
    uniform float uExposure;
    uniform float uFade;
    uniform float uFrame;
    out vec4 outColor;

    vec3 aces(vec3 x) {
      return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
    }
    float hash(vec2 p) {
      p = fract(p * vec2(443.897, 441.423));
      p += dot(p, p.yx + 19.19);
      return fract((p.x + p.y) * p.x);
    }
    void main() {
      vec3 scene = texture(uScene, vUv).rgb;
      vec3 bloom = texture(uBloom, vUv).rgb;
      bloom += texture(uStreak, vUv).rgb * vec3(1.0, 0.5, 0.55) * 0.09;
      bloom += texture(uRays, vUv).rgb * uRayStrength;
      vec2 p = (vUv - 0.5) * uAspect;
      vec2 q = (vUv - uCenter) * uAspect;

      // Deep, warm darkness with a faint crimson breath behind the heart.
      float d2 = dot(q, q);
      vec3 bg = vec3(0.0016, 0.0003, 0.0006)
              + vec3(0.010, 0.0006, 0.0016) * exp(-d2 * 3.0) * (0.8 + 0.4 * uBeat)
              + vec3(0.003, 0.0002, 0.0007) * exp(-d2 * 0.8);

      vec3 c = bg + (scene + bloom * uBloomStrength * (1.0 + 0.35 * uBeat)) * uExposure;
      c *= mix(1.0, 0.35, smoothstep(0.35, 1.25, length(p)));
      c = aces(c * uFade);
      c = pow(c, vec3(1.0 / 2.2));
      // Triangular dither removes banding in the dark gradients.
      float n = hash(gl_FragCoord.xy + fract(uFrame * 0.618) * 97.0)
              + hash(gl_FragCoord.yx * 1.37 + 11.0) - 1.0;
      c += n / 255.0;
      outColor = vec4(c, 1.0);
    }
  `;

  /* ------------------------------------------------------------------ *
   * GL resources                                                       *
   * ------------------------------------------------------------------ */

  const coarse = matchMedia('(pointer: coarse)').matches;
  const small = Math.min(screen.width, screen.height) < 600;
  const mobile = coarse || small;
  const TOTAL = mobile ? 42000 : 90000;
  const FX = mobile
    ? { glint: 160, comet: 2520, shock: 3600, spark: 1100, disc: 9000, bokeh: 26 }
    : { glint: 260, comet: 4800, shock: 6000, spark: 2000, disc: 18000, bokeh: 40 };
  const FX_TOTAL = FX.glint + FX.comet + FX.shock + FX.spark + FX.disc + FX.bokeh;
  const particles = buildParticles(TOTAL, FX);

  let programs, vao, quadVao, particleBuffer, maxPointSize = 64;
  let hdr = { internal: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE, float: false };
  let scene = null, mips = [], streak = null, rays = null;
  let lost = false, failed = false;

  function compile(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS) && !gl.isContextLost()) {
      throw new Error(gl.getShaderInfoLog(shader) || 'shader compile failed');
    }
    return shader;
  }

  function program(vs, fs) {
    const p = gl.createProgram();
    const v = compile(gl.VERTEX_SHADER, vs), f = compile(gl.FRAGMENT_SHADER, fs);
    gl.attachShader(p, v); gl.attachShader(p, f);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS) && !gl.isContextLost()) {
      throw new Error(gl.getProgramInfoLog(p) || 'program link failed');
    }
    gl.deleteShader(v); gl.deleteShader(f);
    const uniforms = {};
    const count = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS) || 0;
    for (let i = 0; i < count; i++) {
      const info = gl.getActiveUniform(p, i);
      uniforms[info.name] = gl.getUniformLocation(p, info.name);
    }
    return { p, u: uniforms };
  }

  function initGL() {
    if (gl.getExtension('EXT_color_buffer_float')) {
      hdr = { internal: gl.RGBA16F, format: gl.RGBA, type: gl.HALF_FLOAT, float: true };
    } else if (gl.getExtension('EXT_color_buffer_half_float')) {
      hdr = { internal: gl.RGBA16F, format: gl.RGBA, type: gl.HALF_FLOAT, float: true };
    }
    gl.getExtension('EXT_float_blend');

    programs = {
      particles: program(PARTICLE_VS, PARTICLE_FS),
      down: program(QUAD_VS, DOWN_FS),
      up: program(QUAD_VS, UP_FS),
      streak: program(QUAD_VS, STREAK_FS),
      rays: program(QUAD_VS, RAYS_FS),
      composite: program(QUAD_VS, COMPOSITE_FS)
    };

    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    particleBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, particleBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, particles, gl.STATIC_DRAW);
    const bytes = STRIDE * 4;
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, bytes, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, bytes, 12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 4, gl.FLOAT, false, bytes, 24);
    gl.bindVertexArray(null);
    quadVao = gl.createVertexArray();

    maxPointSize = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1] || 64;
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    scene = null; mips = []; streak = null; rays = null;
    targetsDirty = true;
  }

  function makeTarget(w, h) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, hdr.internal, w, h, 0, hdr.format, hdr.type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { tex, fbo, w, h, ok };
  }

  function freeTarget(t) {
    if (!t) return;
    gl.deleteTexture(t.tex);
    gl.deleteFramebuffer(t.fbo);
  }

  function buildTargets() {
    freeTarget(scene);
    freeTarget(streak);
    freeTarget(rays);
    mips.forEach(freeTarget);
    mips = [];
    scene = makeTarget(canvas.width, canvas.height);
    if (!scene.ok && hdr.float) {
      // Half-float not renderable after all: fall back to 8-bit.
      freeTarget(scene);
      hdr = { internal: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE, float: false };
      scene = makeTarget(canvas.width, canvas.height);
    }
    let w = canvas.width, h = canvas.height;
    for (let i = 0; i < 7; i++) {
      w = Math.max(1, w >> 1); h = Math.max(1, h >> 1);
      mips.push(makeTarget(w, h));
      if (Math.min(w, h) <= 12) break;
    }
    const src = mips[Math.min(1, mips.length - 1)];
    streak = makeTarget(src.w, src.h);
    rays = makeTarget(src.w, src.h);
    targetsDirty = false;
  }

  /* ------------------------------------------------------------------ *
   * Camera, sizing and interaction                                     *
   * ------------------------------------------------------------------ */

  const FOV = 34 * Math.PI / 180;
  let width = 1, height = 1, dpr = 1, renderScale = mobile ? 0.9 : 1;
  let drawCount = TOTAL;
  let targetsDirty = true, sizeDirty = true;
  let camDist = 6, focal = 1;
  const viewProj = new Float32Array(16);
  const model = new Float32Array(9);
  const camera = [0, 0, 6];

  function resize() {
    width = Math.max(1, innerWidth);
    height = Math.max(1, innerHeight);
    dpr = Math.min(window.devicePixelRatio || 1, 2) * renderScale;
    const w = Math.max(1, Math.round(width * dpr));
    const h = Math.max(1, Math.round(height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
      targetsDirty = true;
    }
    const aspect = width / height;
    const tanHalf = Math.tan(FOV / 2);
    // Frame the heart (≈1.2 half-height, 1.2 half-width) with breathing room.
    const fillH = 0.64, fillW = 0.8;
    camDist = Math.max(1.2 / (tanHalf * fillH), 1.2 / (tanHalf * aspect * fillW)) + 0.7;
    focal = 0.5 * canvas.height / tanHalf;
    camera[2] = camDist;

    const f = 1 / tanHalf, near = 0.1, far = 60;
    const nf = 1 / (near - far);
    const proj = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
    // view = translate(0, 0, -camDist)
    for (let i = 0; i < 16; i++) viewProj[i] = proj[i];
    for (let r = 0; r < 4; r++) viewProj[12 + r] = proj[8 + r] * -camDist + proj[12 + r];
    sizeDirty = false;
  }

  const pointer = { x: 0, y: 0, inside: false, down: false, id: -1, lastX: 0, lastY: 0, lastT: 0, moved: 0, presence: 0 };
  const orbit = { yaw: 0, pitch: 0, vyaw: 0, vpitch: 0, hoverPitch: 0 };
  const REVOLUTION = 9; // seconds per full turn
  const ray = [0, 0, -1];
  const ripple = { dir: [0, 0, -1], age: 9 };

  function pointerRay(clientX, clientY, out) {
    const nx = (clientX / width) * 2 - 1, ny = 1 - (clientY / height) * 2;
    const tanHalf = Math.tan(FOV / 2);
    const x = nx * tanHalf * (width / height), y = ny * tanHalf, z = -1;
    const l = Math.hypot(x, y, z);
    out[0] = x / l; out[1] = y / l; out[2] = z / l;
  }

  canvas.addEventListener('pointerdown', e => {
    pointer.down = true; pointer.id = e.pointerId;
    pointer.x = pointer.lastX = e.clientX; pointer.y = pointer.lastY = e.clientY;
    pointer.inside = true; pointer.moved = 0; pointer.lastT = e.timeStamp;
    orbit.vyaw = orbit.vpitch = 0;
    try { canvas.setPointerCapture(e.pointerId); } catch { /* not supported */ }
  });
  canvas.addEventListener('pointermove', e => {
    pointer.x = e.clientX; pointer.y = e.clientY; pointer.inside = true;
    if (pointer.down && e.pointerId === pointer.id) {
      const dx = e.clientX - pointer.lastX, dy = e.clientY - pointer.lastY;
      pointer.moved += Math.abs(dx) + Math.abs(dy);
      const k = 3.2 / Math.min(width, height);
      orbit.yaw += dx * k; orbit.pitch += dy * k * 0.6;
      // Fling velocity from real event timing (smoothed), not frame count.
      const ms = Math.max(8, Math.min(100, e.timeStamp - pointer.lastT));
      orbit.vyaw = orbit.vyaw * 0.4 + (dx * k * 1000 / ms) * 0.6;
      orbit.vpitch = orbit.vpitch * 0.4 + (dy * k * 600 / ms) * 0.6;
      pointer.lastX = e.clientX; pointer.lastY = e.clientY; pointer.lastT = e.timeStamp;
    }
  }, { passive: true });
  const release = e => {
    if (e.pointerId !== pointer.id) return;
    pointer.down = false; pointer.id = -1;
    if (e.pointerType !== 'mouse') pointer.inside = false;
    if (e.type === 'pointerup' && pointer.moved < 10) tap(e.clientX, e.clientY);
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('pointerleave', e => { if (e.pointerType === 'mouse') pointer.inside = false; });
  window.addEventListener('blur', () => { pointer.inside = false; pointer.down = false; });

  function tap(x, y) {
    audio.enable();
    if (reduced()) return;
    pointerRay(x, y, ripple.dir);
    ripple.age = 0;
  }

  window.addEventListener('keydown', e => {
    if (e.repeat || e.ctrlKey || e.altKey || e.metaKey) return;
    if (e.code === 'KeyM') audio.toggleMuted();
    else if (e.code === 'Space' || e.code === 'Enter') {
      e.preventDefault();
      audio.enable();
    }
  });

  /* ------------------------------------------------------------------ *
   * Frame loop                                                         *
   * ------------------------------------------------------------------ */

  const reduced = () => motionQuery.matches;
  let time = 0, intro = 0, last = 0, raf = 0, frameIndex = 0;
  let slowTime = 0, budgetSteps = 0;

  function damp(current, target, rate, dt) {
    return target + (current - target) * Math.exp(-rate * dt);
  }

  function update(dt) {
    time += dt;
    intro = Math.min(1, intro + dt / 3.4);
    rhythm.update(time);
    audio.schedule(rhythm, time);

    // Drag inertia. A drag shifts the phase of the turntable rotation (it
    // is kept, so the spin simply carries on from there); tilt eases back.
    if (!pointer.down) {
      orbit.yaw += orbit.vyaw * dt; orbit.pitch += orbit.vpitch * dt;
      orbit.vyaw = damp(orbit.vyaw, 0, 3.5, dt); orbit.vpitch = damp(orbit.vpitch, 0, 3.5, dt);
      orbit.pitch = damp(orbit.pitch, 0, 0.9, dt);
    }
    orbit.yaw %= Math.PI * 2;
    orbit.pitch = Math.max(-0.7, Math.min(0.7, orbit.pitch));
    const hover = pointer.inside && !pointer.down && !coarse;
    orbit.hoverPitch = damp(orbit.hoverPitch, hover ? (pointer.y / height - 0.5) * 0.22 : 0, 2.2, dt);
    pointer.presence = damp(pointer.presence, pointer.inside ? 1 : 0, 4, dt);
    if (pointer.presence > 0.001) pointerRay(pointer.x, pointer.y, ray);
    ripple.age += dt;
  }

  function setModel(t) {
    // Turntable: one full revolution about the vertical axis every
    // REVOLUTION seconds at a perfectly constant angular speed. Hover only
    // tilts (pitch), so it never speeds up or slows down the spin.
    const yaw = ((t % REVOLUTION) / REVOLUTION) * Math.PI * 2 + orbit.yaw;
    const pitch = -0.08 + orbit.pitch + orbit.hoverPitch;
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    // R = Rx(pitch) * Ry(yaw), column-major.
    model[0] = cy;        model[1] = sp * sy;  model[2] = -cp * sy;
    model[3] = 0;         model[4] = cp;       model[5] = sp;
    model[6] = sy;        model[7] = -sp * cy; model[8] = cp * cy;
  }

  function quad(prog, target) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fbo : null);
    gl.viewport(0, 0, target ? target.w : canvas.width, target ? target.h : canvas.height);
    gl.useProgram(prog.p);
    gl.bindVertexArray(quadVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  function render() {
    if (lost || failed) return;
    if (sizeDirty) resize();
    if (targetsDirty) buildTargets();
    const t = reduced() ? 12.0 : time;
    setModel(reduced() ? 0 : time); // still frame faces the viewer
    const beatAges = reduced() ? [10, 10] : [time - rhythm.current, time - rhythm.previous];
    const globalBeat = beatShape(beatAges[0]) + beatShape(beatAges[1]);
    const bob = Math.sin(t * 0.37) * 0.025;

    // 1. Particles → HDR scene.
    const P = programs.particles, u = P.u;
    gl.bindFramebuffer(gl.FRAMEBUFFER, scene.fbo);
    gl.viewport(0, 0, scene.w, scene.h);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(P.p);
    gl.uniformMatrix4fv(u.uViewProj, false, viewProj);
    gl.uniformMatrix3fv(u.uModel, false, model);
    gl.uniform3f(u.uOffset, 0, bob, 0);
    gl.uniform3f(u.uCamera, camera[0], camera[1], camera[2]);
    gl.uniform1f(u.uTime, t);
    gl.uniform2f(u.uBeatAge, beatAges[0], beatAges[1]);
    gl.uniform1f(u.uIntro, reduced() ? 1 : intro);
    gl.uniform1f(u.uFocal, focal);
    gl.uniform1f(u.uFocus, camDist);
    const tanHalf = Math.tan(FOV / 2);
    gl.uniform2f(u.uFrustum, tanHalf * width / height, tanHalf);
    gl.uniform1f(u.uDiscGain, mobile ? 1.6 : 1);
    gl.uniform1f(u.uMaxSize, Math.min(maxPointSize, 96));
    // Fewer particles (mobile, adaptive quality) are drawn slightly larger
    // and brighter so the image keeps the same density and energy.
    const ratio = 90000 / drawCount;
    gl.uniform1f(u.uSize, 0.0105 * Math.pow(ratio, 0.33));
    gl.uniform1f(u.uGain, (hdr.float ? 1 : 0.55) * Math.pow(ratio, 0.34) * 0.62);
    gl.uniform4f(u.uRay, ray[0], ray[1], ray[2], reduced() ? 0 : pointer.presence);
    gl.uniform4f(u.uRipple, ripple.dir[0], ripple.dir[1], ripple.dir[2], ripple.age);
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.POINTS, 0, drawCount);
    // Effect layers: fixed counts, drawn at their designed size and energy.
    gl.uniform1f(u.uSize, 0.0105);
    gl.uniform1f(u.uGain, (hdr.float ? 1 : 0.55) * 0.62);
    gl.drawArrays(gl.POINTS, TOTAL, FX_TOTAL);
    gl.disable(gl.BLEND);

    // 2. Bloom: downsample chain, then additive upsample.
    const D = programs.down;
    gl.useProgram(D.p);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(D.u.uTex, 0);
    let src = scene;
    for (let i = 0; i < mips.length; i++) {
      gl.useProgram(D.p);
      gl.bindTexture(gl.TEXTURE_2D, src.tex);
      gl.uniform2f(D.u.uTexel, 1 / src.w, 1 / src.h);
      gl.uniform1f(D.u.uPrefilter, i === 0 ? 1 : 0);
      quad(D, mips[i]);
      src = mips[i];
    }
    const U = programs.up;
    gl.useProgram(U.p);
    gl.uniform1i(U.u.uTex, 0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    for (let i = mips.length - 1; i > 0; i--) {
      gl.useProgram(U.p);
      gl.bindTexture(gl.TEXTURE_2D, mips[i].tex);
      gl.uniform2f(U.u.uTexel, 1 / mips[i].w, 1 / mips[i].h);
      gl.uniform1f(U.u.uWeight, i > 2 ? 0.6 : 0.9);
      quad(U, mips[i - 1]);
    }
    gl.disable(gl.BLEND);

    // Streak from the quarter-resolution bloom level.
    const S = programs.streak;
    const streakSrc = mips[Math.min(1, mips.length - 1)];
    gl.useProgram(S.p);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, streakSrc.tex);
    gl.uniform1i(S.u.uTex, 0);
    gl.uniform1f(S.u.uStep, 1.8 / streakSrc.w);
    quad(S, streak);

    const Y = programs.rays;
    gl.useProgram(Y.p);
    gl.uniform1i(Y.u.uTex, 0);
    gl.uniform2f(Y.u.uCenter, 0.5, 0.5 + bob * 0.1);
    quad(Y, rays);

    // 3. Composite.
    const C = programs.composite;
    gl.useProgram(C.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, scene.tex);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, mips[0].tex);
    gl.uniform1i(C.u.uScene, 0);
    gl.uniform1i(C.u.uBloom, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, streak.tex);
    gl.uniform1i(C.u.uStreak, 2);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, rays.tex);
    gl.uniform1i(C.u.uRays, 3);
    gl.uniform1f(C.u.uRayStrength, 0.16 * (1 + 0.6 * (reduced() ? 0 : globalBeat)));
    const aspect = width / height;
    gl.uniform2f(C.u.uAspect, aspect >= 1 ? aspect : 1, aspect >= 1 ? 1 : 1 / aspect);
    gl.uniform2f(C.u.uCenter, 0.5, 0.5 + bob * 0.1);
    gl.uniform1f(C.u.uBeat, reduced() ? 0 : globalBeat);
    gl.uniform1f(C.u.uBloomStrength, 0.75);
    gl.uniform1f(C.u.uExposure, 1.15);
    gl.uniform1f(C.u.uFade, reduced() ? 1 : Math.min(1, intro * 2.5));
    gl.uniform1f(C.u.uFrame, frameIndex++);
    quad(C, null);
    gl.activeTexture(gl.TEXTURE0);
  }

  function adapt(dt) {
    // Below ~45 fps for a sustained stretch: trade resolution, then
    // particle count, for smoothness. Never oscillates back up.
    if (time < 3.5 || budgetSteps >= 6) return;
    if (dt > 1 / 45) slowTime += dt;
    else slowTime = Math.max(0, slowTime - dt * 0.5);
    if (slowTime < 1.5) return;
    slowTime = 0;
    budgetSteps++;
    if (renderScale > 0.62) { renderScale *= 0.85; sizeDirty = true; }
    else drawCount = Math.max(Math.floor(TOTAL * 0.4), Math.floor(drawCount * 0.8));
  }

  function frame(now) {
    raf = requestAnimationFrame(frame);
    const dt = last ? Math.min(Math.max((now - last) / 1000, 0), 1 / 15) : 1 / 60;
    last = now;
    update(dt);
    render();
    adapt(dt);
  }

  function start() {
    if (raf || lost || failed || document.hidden || reduced()) return;
    last = 0;
    raf = requestAnimationFrame(frame);
  }
  function stop() {
    cancelAnimationFrame(raf);
    raf = 0;
  }
  function syncActivity() {
    const active = !document.hidden && !lost && !reduced();
    audio.setActive(active);
    if (active) start();
    else { stop(); if (!lost && !document.hidden) render(); }
  }

  window.addEventListener('resize', () => { sizeDirty = true; if (!raf) render(); });
  document.addEventListener('visibilitychange', syncActivity);
  motionQuery.addEventListener('change', syncActivity);
  window.addEventListener('pageshow', syncActivity);
  window.addEventListener('pagehide', e => { if (!e.persisted) audio.dispose(); else audio.setActive(false); });
  canvas.addEventListener('webglcontextlost', e => {
    e.preventDefault();
    lost = true;
    stop();
  });
  canvas.addEventListener('webglcontextrestored', () => {
    lost = false;
    initGL();
    sizeDirty = true;
    syncActivity();
  });

  try {
    initGL();
  } catch (err) {
    console.error(err);
    failed = true;
    stop();
    return startFallback();
  }
  root.classList.add('ready');
  resize();
  render();
  syncActivity();

  // Test / debugging hook: deterministic rendering at a given time.
  window.__heart = {
    renderAt(t) { stop(); time = t; intro = Math.min(1, t / 3.4); rhythm.update(t); render(); },
    get stats() {
      return { drawCount, renderScale, hdr: hdr.float, total: TOTAL, audio: audio.ctx ? audio.ctx.state : 'off', scheduled: audio.scheduled.size };
    }
  };

  /* ------------------------------------------------------------------ *
   * Fallback (no WebGL2): a CSS heart that keeps the same rhythm.      *
   * ------------------------------------------------------------------ */
  function startFallback() {
    root.classList.add('no-webgl');
    const heart = document.querySelector('.fallback-heart');
    if (!heart) return;
    let t = 0, prev = 0;
    const loop = now => {
      const dt = prev ? Math.min((now - prev) / 1000, 0.1) : 0;
      prev = now;
      t += dt;
      rhythm.update(t);
      audio.schedule(rhythm, t);
      const b = beatShape(t - rhythm.current) + beatShape(t - rhythm.previous);
      heart.style.transform = `scale(${1 + 0.07 * b})`;
      heart.style.filter = `drop-shadow(0 0 ${18 + 22 * b}px rgba(220, 20, 50, ${0.45 + 0.3 * b}))`;
      if (!motionQuery.matches) requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
    document.addEventListener('pointerup', () => audio.enable());
  }
})();
