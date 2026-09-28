# Heart

Pure static particle-heart webpage with synchronized heartbeat audio.

## Files

- `index.html`
- `style.css`
- `app.js`
- `heartbeat.js`
- `docs/nginx.conf.example`
- `docs/部署说明.txt`

## Deploy on a VPS

```bash
git clone https://github.com/VAR7IANT/Heart.git
sudo mkdir -p /var/www/heart
sudo cp -r Heart/index.html Heart/style.css Heart/app.js Heart/heartbeat.js /var/www/heart/
```

Point Nginx at `/var/www/heart`, then bind your domain and enable HTTPS.

The site has no build step and no Node.js dependency.
