# Heart

A cinematic, GPU-driven 3D particle heart with a realistic lub-dub heartbeat.
It's a static site with no build step and no dependencies, and it works offline.

## Files

- `index.html`: page shell and a CSS fallback heart
- `style.css`: full-bleed canvas and the fallback styles
- `app.js`: WebGL2 renderer (geometry, shaders, bloom, interaction)
- `heartbeat.js`: heart rhythm and synthesized heartbeat audio
- `docs/nginx.conf.example`, `docs/部署说明.txt`: deployment notes

## How it works

- **Geometry**: about 90k particles (42k on phones) are generated once, on and
  inside Taubin's implicit heart surface. The surface is written in its
  well-conditioned cube-root form, `g − z·∛(x² + 9/80·y²) = 0`. The classic
  sextic's gradient vanishes along the equator, and that is what causes the
  bright horizontal seam in naive renderings. Surface samples are kept by
  true distance to the surface, so density stays uniform with no knots or
  seams.
- **Motion runs on the GPU**: the particle buffer is static. The vertex shader
  streams surface particles along the skin through a divergence-free flow
  field and snaps them back onto the surface analytically. It also advects the
  aura outward, and it applies the heartbeat as a pressure wave that travels
  from the core to the skin.
- **Look**: Fresnel rim lighting gives a dark interior and luminous edges.
  Particles blend additively into a half-float HDR target, so nothing clips or
  flickers. Sub-pixel points are widened with energy conservation, which stops
  shimmer. Distant embers get depth-of-field blur. A dual-filter bloom, ACES
  tone mapping, a vignette and dithering finish the image.
- **Heartbeat**: a deterministic rhythm (about 62 BPM, with gentle natural
  variability) drives both the visuals and Web Audio. Each beat is a lub (S1)
  followed by a softer dub (S2) about 0.3 s later. Sound is synthesized and
  scheduled ahead on the audio clock, compensated for output latency, so it
  never drifts.
- **Smoothness**: there is no per-frame CPU work proportional to the particle
  count and no frame skipping. If the frame rate stays below about 45 fps, the
  renderer first lowers its resolution and then the particle count, and the
  image keeps the same brightness and density.

## Interaction

- Tap or click to start the heartbeat sound and send a ripple through the
  particles.
- Drag to rotate the heart. It eases back to its resting pose.
- On desktop, hovering tilts the heart slightly and particles part around the
  cursor.
- `Space` or `Enter` starts the sound, and `M` mutes it.
- Rendering pauses in background tabs. With `prefers-reduced-motion`, a still
  frame is shown. Browsers without WebGL2 get a CSS heart that beats in the
  same rhythm.

## Deploy on a VPS

```bash
git clone https://github.com/VAR7IANT/Heart.git
sudo mkdir -p /var/www/heart
sudo cp Heart/index.html Heart/style.css Heart/app.js Heart/heartbeat.js /var/www/heart/
```

Point Nginx at `/var/www/heart` (see `docs/nginx.conf.example`), then bind
your domain and enable HTTPS.
