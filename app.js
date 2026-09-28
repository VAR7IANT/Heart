(() => {
  'use strict';

  let canvas = document.getElementById('universe');
  let gl = canvas.getContext('webgl', { alpha: true, antialias: false, depth: false, stencil: false });
  let ctx, program, buffer, maxPointSize = 96, contextLost = false;
  const motionPreference = matchMedia('(prefers-reduced-motion: reduce)');
  const heartbeat = new window.HeartbeatAudio();
  const TAU = Math.PI * 2;
  const compact = innerWidth < 700;
  let width = innerWidth, height = innerHeight, dpr = 1;
  let time = 0, previous = 0, raf = 0, burstAt = -20;
  let pointerX = width / 2, pointerY = height / 2, pointerActive = false;
  let tiltX = 0, tiltY = 0, lastPointerSpark = 0;
  const hearts = [], field = [], halo = [], streams = [], meteors = [], sparks = [];
  const counts = gl
    ? { heart: compact ? 6800 : 11800, field: compact ? 1500 : 3100, halo: compact ? 1000 : 2000, stream: compact ? 700 : 1400 }
    : { heart: 2800, field: 550, halo: 300, stream: 250 };
  const capacity = counts.heart * 2 + counts.field + counts.halo + counts.stream + 1800;
  const vertices = new Float32Array(capacity * 8);
  let used = 0;
  const random = (a, b) => a + Math.random() * (b - a);
  const wrap = value => ((value % 1) + 1) % 1;

  function heartShape(t) {
    return [16 * Math.sin(t) ** 3, -(13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)) + 2];
  }

  for (let i = 0; i < counts.heart; i++) {
    const t = random(0, TAU), [x, y] = heartShape(t);
    const shell = i < counts.heart * .28;
    const r = shell ? random(.94, 1.025) : Math.sqrt(Math.random()) * .97;
    hearts.push({ x: x * r, y: (y - 4) * r + 4, z: random(-5, 5) * Math.sqrt(Math.max(0, 1 - r * r)),
      t, r, phase: random(0, TAU), size: random(.8, 2), light: Math.random(),
      vx: random(-65, 65), vy: random(-55, 55), vz: random(-20, 20) });
  }
  for (let i = 0; i < counts.field; i++) {
    field.push({ x: Math.random(), y: Math.random(), depth: Math.random(), phase: random(0, TAU),
      speed: random(.002, .013), size: random(.7, 2.5), tint: Math.random(), bright: Math.random() });
  }
  for (let i = 0; i < counts.halo; i++) {
    const t = random(0, TAU), [x, y] = heartShape(t), r = random(1.015, 1.33);
    halo.push({ x: x * r + random(-1, 1), y: (y - 4) * r + 4 + random(-1, 1),
      phase: random(0, TAU), radius: r, size: random(.8, 2.5), light: Math.random() });
  }
  for (let i = 0; i < counts.stream; i++) {
    streams.push({ u: Math.random(), spread: random(-1, 1) * Math.random(), phase: random(0, TAU),
      size: random(.6, 2), side: i % 2 });
  }
  for (let i = 0; i < (compact ? 3 : 5); i++) {
    meteors.push({ offset: i * 2.4 + Math.random(), period: random(9, 16), x: Math.random(), y: random(.02, .7), length: random(100, 220) });
  }

  function initializeGL() {
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error('Shader compilation failed');
      return shader;
    };
    const vertex = compile(gl.VERTEX_SHADER, `
      attribute vec2 aPosition;
      attribute float aSize;
      attribute vec3 aColor;
      attribute float aOpacity;
      attribute float aKind;
      varying vec3 vColor;
      varying float vOpacity;
      varying float vKind;
      void main() {
        gl_Position = vec4(aPosition, 0.0, 1.0);
        gl_PointSize = aSize;
        vColor = aColor; vOpacity = aOpacity; vKind = aKind;
      }
    `);
    const fragment = compile(gl.FRAGMENT_SHADER, `
      precision mediump float;
      varying vec3 vColor;
      varying float vOpacity;
      varying float vKind;
      void main() {
        vec2 p = gl_PointCoord * 2.0 - 1.0;
        float r = length(p);
        if (r > 1.0) discard;
        float edge = 1.0 - smoothstep(0.65, 1.0, r);
        float light = (exp(-r*r*5.5)*0.65 + exp(-r*r*32.0)*0.65)*edge;
        if (vKind > 0.5 && vKind < 1.5) light = exp(-r*r*3.8)*edge*0.5;
        if (vKind > 1.5) {
          float rays = pow(max(0.0,1.0-abs(p.x)),22.0)*pow(max(0.0,1.0-abs(p.y)),1.6)
                     + pow(max(0.0,1.0-abs(p.y)),22.0)*pow(max(0.0,1.0-abs(p.x)),1.6);
          light = (exp(-r*r*24.0) + rays*0.6)*edge;
        }
        gl_FragColor = vec4(vColor, min(1.0, light*vOpacity));
      }
    `);
    program = gl.createProgram(); gl.attachShader(program, vertex); gl.attachShader(program, fragment);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error('Shader link failed');
    gl.deleteShader(vertex); gl.deleteShader(fragment); gl.useProgram(program);
    buffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, vertices.byteLength, gl.DYNAMIC_DRAW);
    for (const [name, size, offset] of [['aPosition', 2, 0], ['aSize', 1, 8], ['aColor', 3, 12], ['aOpacity', 1, 24], ['aKind', 1, 28]]) {
      const attribute = gl.getAttribLocation(program, name);
      gl.enableVertexAttribArray(attribute); gl.vertexAttribPointer(attribute, size, gl.FLOAT, false, 32, offset);
    }
    maxPointSize = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1];
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE); gl.clearColor(0, 0, 0, 0);
    gl.viewport(0, 0, canvas.width, canvas.height);
  }

  if (gl) {
    try { initializeGL(); }
    catch {
      const replacement = canvas.cloneNode(false); canvas.replaceWith(replacement); canvas = replacement;
      gl = null; ctx = canvas.getContext('2d');
    }
  } else ctx = canvas.getContext('2d');

  function point(x, y, size, red, green, blue, opacity = 1, kind = 0) {
    if (used >= capacity || opacity < .008 || x < -100 || x > width + 100 || y < -100 || y > height + 100) return;
    const n = used++ * 8;
    vertices[n] = x / width * 2 - 1; vertices[n + 1] = 1 - y / height * 2;
    vertices[n + 2] = Math.max(1, Math.min(maxPointSize, size * dpr));
    vertices[n + 3] = red; vertices[n + 4] = green; vertices[n + 5] = blue;
    vertices[n + 6] = opacity; vertices[n + 7] = kind;
  }

  function draw() {
    if (contextLost) return;
    used = 0;
    const scale = Math.min(width * .0245, height * .0245), cx = width / 2, cy = height / 2 - 4.5 * scale;
    const elapsed = time - burstAt;
    const burst = elapsed >= 0 && elapsed < 4.2 ? Math.sin(Math.PI * elapsed / 4.2) ** 2 * Math.exp(-elapsed * .36) : 0;
    const pulsePhase = heartbeat.phase(time);
    const pulse = Math.exp(-(((pulsePhase - .12) / .09) ** 2)) + .48 * Math.exp(-(((pulsePhase - .32) / .08) ** 2));
    const beat = 1 + pulse * .034;
    const ay = Math.sin(time * .24) * .17 + tiltY, ax = tiltX;
    const co = Math.cos(ay), si = Math.sin(ay), coa = Math.cos(ax), sia = Math.sin(ax);

    // A full-screen field at three depths keeps the edges alive.
    for (const p of field) {
      const x = wrap(p.x + Math.sin(time * .07 + p.phase) * .016 + time * p.speed * .12) * width;
      const y = wrap(p.y - time * p.speed * .32) * height;
      const twinkle = .3 + .7 * (.5 + .5 * Math.sin(time * (1 + p.depth) + p.phase)) ** 4;
      const front = p.depth > .92;
      const size = front ? 5 + p.size * 4 : p.size * (1 + p.depth);
      const sideLight = .65 + .35 * Math.min(1, Math.abs(x - cx) / (width * .32));
      const alpha = (front ? .2 : .27 + p.depth * .32) * twinkle * sideLight;
      point(x, y, size, .65 + p.tint * .35, .32 + p.tint * .22, .78 + p.tint * .18, alpha, front ? 1 : 0);
      if (p.bright > .984) point(x, y, 11 + twinkle * 9, .86, .73, 1, twinkle * .8, 2);
    }

    // Two broad, softly scattered streams curl behind the heart.
    for (const p of streams) {
      const u = wrap(p.u + time * .009 * (p.side ? -1 : 1));
      const x = u * width * 1.2 - width * .1;
      const wave = Math.sin(u * TAU * .85 + p.side * Math.PI + time * .05);
      const y = height * (.5 + wave * .23) + p.spread * height * .115;
      const alpha = (.15 + .26 * (1 - Math.abs(p.spread))) * (.7 + .3 * Math.sin(p.phase + time));
      point(x, y, p.size * 1.7, p.side ? .56 : 1, .25, p.side ? 1 : .58, alpha);
    }

    for (const p of halo) {
      const drift = 1 + .024 * Math.sin(time * .7 + p.phase);
      const x = cx + (p.x * drift * co + Math.sin(p.phase + time * .3)) * scale;
      const y = cy + (p.y * drift + Math.cos(p.phase + time * .4) * .7) * scale;
      const alpha = (1.45 - p.radius) * (.35 + p.light * .5) * (1 + pulse * .3);
      point(x, y, p.size * 2, 1, .19 + p.light * .2, .5 + p.light * .25, alpha);
    }

    // A moving highlight wraps around the heart's volume and its bright rim.
    for (let i = 0; i < hearts.length; i++) {
      const p = hearts[i], wave = .5 + .5 * Math.sin(p.t * 3 - time * 1.6 + p.r * 4);
      let x = p.x * beat + p.vx * burst + Math.sin(time * .7 + p.phase) * .12;
      let y = (p.y - 4) * beat + 4 + p.vy * burst + Math.cos(time * .6 + p.phase) * .12;
      const z = p.z + p.vz * burst;
      const rx = x * co + z * si, rz = -x * si + z * co;
      const ry = y * coa - rz * sia, depth = y * sia + rz * coa, perspective = 85 / (85 - depth);
      x = cx + rx * scale * perspective; y = cy + ry * scale * perspective;
      let touch = 0;
      if (pointerActive && !motionPreference.matches) {
        const dx = x - pointerX, dy = y - pointerY, distance = Math.hypot(dx, dy);
        if (distance < 130 && distance > 1) {
          touch = (1 - distance / 130) ** 2;
          x += dx / distance * touch * 19; y += dy / distance * touch * 19;
        }
      }
      const shine = p.light > .974;
      const light = (.38 + p.light * .48) * (.75 + wave * .3) * (1 + depth * .035 + touch * .8);
      const green = .14 + wave * .2 + (shine ? .43 : 0);
      const blue = .38 + wave * .25 + (shine ? .24 : 0);
      point(x, y, p.size * perspective * (shine ? 3.5 : 2.1), 1, green, blue, light);
      if (i % 9 === 0) point(x, y, (15 + p.size * 8) * perspective, 1, .11, .4, light * .16, 1);
      if (p.light > .995) point(x, y, 12 + wave * 7, 1, .73, .85, light * .85, 2);
    }

    // Independent schedules produce occasional diagonal shooting stars.
    for (const m of meteors) {
      const age = (time + m.offset) % m.period;
      if (age > 1.7) continue;
      const cycle = Math.floor((time + m.offset) / m.period);
      const x = wrap(m.x + cycle * .3819) * width + age * width * .3;
      const y = wrap(m.y + cycle * .217) * height * .75 + age * height * .19;
      const fade = Math.sin(age / 1.7 * Math.PI);
      for (let j = 0; j < 40; j++) {
        const f = j / 40;
        point(x - f * m.length, y - f * m.length * .52, 1.5 + (1 - f) * 2.5, .92, .63, 1, (1 - f) ** 2 * fade * .75);
      }
      point(x, y, 13, 1, .85, 1, fade, 2);
    }

    // A click sends a wave through the surrounding dust as the heart opens.
    if (elapsed >= 0 && elapsed < 2.8) {
      const radius = elapsed * Math.max(width, height) * .43;
      const alpha = Math.max(0, 1 - elapsed / 2.8) ** 2;
      for (let i = 0; i < 480; i++) {
        const angle = i / 480 * TAU, scatter = Math.sin(i * 34.7) * 9;
        point(cx + Math.cos(angle) * (radius + scatter), height / 2 + Math.sin(angle) * (radius + scatter), 2.3, 1, .37, .7, alpha * .75);
      }
    }
    for (let i = sparks.length - 1; i >= 0; i--) {
      const p = sparks[i], age = time - p.born;
      if (age > 1.8) { sparks.splice(i, 1); continue; }
      const fade = (1 - age / 1.8) ** 2;
      point(p.x + p.vx * age, p.y + p.vy * age + age * age * 15, p.size * (1 + age), 1, .55, .8, fade, p.star ? 2 : 0);
    }

    if (gl) {
      gl.clear(gl.COLOR_BUFFER_BIT); gl.bufferSubData(gl.ARRAY_BUFFER, 0, vertices.subarray(0, used * 8)); gl.drawArrays(gl.POINTS, 0, used);
    } else if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.globalCompositeOperation = 'lighter';
      const step = hearts.length > 3000 ? 3 : 1;
      for (let i = 0; i < used; i += step) {
        const n = i * 8, glow = vertices[n + 7] === 1;
        ctx.globalAlpha = Math.min(1, vertices[n + 6] * (glow ? .13 : .8));
        ctx.fillStyle = `rgb(${Math.round(vertices[n + 3] * 255)},${Math.round(vertices[n + 4] * 255)},${Math.round(vertices[n + 5] * 255)})`;
        ctx.beginPath(); ctx.arc((vertices[n] + 1) / 2 * canvas.width, (1 - vertices[n + 1]) / 2 * canvas.height, vertices[n + 2] * .3, 0, TAU); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
  }

  function frame(now) {
    raf = 0;
    if (document.hidden || motionPreference.matches || contextLost) return;
    if (now - previous < 16) { raf = requestAnimationFrame(frame); return; }
    time += Math.min((now - previous) / 1000, .05); previous = now;
    tiltY += ((pointerActive ? (pointerX / width - .5) * .3 : 0) - tiltY) * .04;
    tiltX += ((pointerActive ? (pointerY / height - .5) * .16 : 0) - tiltX) * .04;
    draw(); raf = requestAnimationFrame(frame);
  }
  function resume() {
    if (!raf && !document.hidden && !motionPreference.matches && !contextLost) { previous = performance.now(); raf = requestAnimationFrame(frame); }
  }
  function resize() {
    width = Math.max(1, innerWidth); height = Math.max(1, innerHeight); dpr = Math.min(devicePixelRatio || 1, compact ? 1.6 : 2);
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    if (gl && !contextLost) gl.viewport(0, 0, canvas.width, canvas.height);
    draw();
  }
  canvas.addEventListener('click', () => {
    void heartbeat.enable(time);
    if (!motionPreference.matches && time - burstAt > 1.5) burstAt = time;
  });
  window.addEventListener('keydown', event => {
    if (event.repeat || event.ctrlKey || event.altKey || event.metaKey) return;
    if (event.code === 'KeyM') heartbeat.toggleMuted(time);
    if (event.code === 'Space' || event.code === 'Enter') { event.preventDefault(); void heartbeat.enable(time); }
  });
  window.addEventListener('pointermove', event => {
    pointerX = event.clientX; pointerY = event.clientY; pointerActive = true;
    if (motionPreference.matches || time - lastPointerSpark < .03) return;
    lastPointerSpark = time;
    for (let i = 0; i < 4 && sparks.length < 160; i++) sparks.push({ x: pointerX, y: pointerY, vx: random(-28, 28), vy: random(-35, 15), born: time, size: random(2, 5), star: Math.random() > .8 });
  }, { passive: true });
  document.addEventListener('pointerleave', () => { pointerActive = false; });
  window.addEventListener('pointerup', event => { if (event.pointerType === 'touch') pointerActive = false; });
  window.addEventListener('blur', () => { pointerActive = false; });
  window.addEventListener('resize', resize);
  document.addEventListener('visibilitychange', () => {
    heartbeat.setActive(!document.hidden && !motionPreference.matches && !contextLost);
    if (document.hidden) { cancelAnimationFrame(raf); raf = 0; } else resume();
  });
  motionPreference.addEventListener('change', () => {
    heartbeat.setActive(!document.hidden && !motionPreference.matches && !contextLost);
    if (motionPreference.matches) { cancelAnimationFrame(raf); raf = 0; draw(); } else resume();
  });
  canvas.addEventListener('webglcontextlost', event => { event.preventDefault(); contextLost = true; heartbeat.setActive(false); cancelAnimationFrame(raf); raf = 0; });
  canvas.addEventListener('webglcontextrestored', () => { initializeGL(); contextLost = false; heartbeat.setActive(!document.hidden && !motionPreference.matches); resize(); resume(); });
  window.addEventListener('pagehide', event => { if (event.persisted) heartbeat.setActive(false); else heartbeat.dispose(); });
  window.addEventListener('pageshow', () => heartbeat.setActive(!document.hidden && !motionPreference.matches && !contextLost));
  heartbeat.setActive(!document.hidden && !motionPreference.matches);
  resize(); resume();
})();
