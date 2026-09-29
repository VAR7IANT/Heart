# Heart

Pure static particle-heart webpage with synchronized heartbeat audio.

## Files

- `index.html`
- `style.css`
- `app.js`
- `heartbeat.js`
- `docs/nginx.conf.example`
- `docs/部署说明.txt`
- `docs/audio-attribution.txt`

## Playback

The heart animation runs without a build step or Node.js. On the first click or tap, the browser plays a real heartbeat recording in sync with the animation. Audio streams from Wikimedia Commons, so playback requires an internet connection. Press `M` to mute or unmute.

The recording's creator and CC BY 3.0 license are documented in `docs/audio-attribution.txt`.

## Deploy on a VPS

```bash
git clone https://github.com/VAR7IANT/Heart.git
sudo mkdir -p /var/www/heart
sudo cp -r Heart/index.html Heart/style.css Heart/app.js Heart/heartbeat.js /var/www/heart/
```

Point Nginx at `/var/www/heart`, then bind your domain and enable HTTPS.
