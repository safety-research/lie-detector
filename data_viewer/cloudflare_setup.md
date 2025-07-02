# Cloudflare Tunnel Setup (Secure External Access)

## Step 1: Install Cloudflare Tunnel
```bash
# Download and install cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
```

## Step 2: Authenticate with Cloudflare
```bash
cloudflared tunnel login
```

## Step 3: Create a tunnel
```bash
cloudflared tunnel create lie-detector-viewer
```

## Step 4: Configure the tunnel
Create a config file at `~/.cloudflared/config.yml`:
```yaml
tunnel: <your-tunnel-id>
credentials-file: ~/.cloudflared/<your-tunnel-id>.json

ingress:
  - hostname: lie-detector-viewer.your-domain.com
    service: http://localhost:9009
  - service: http_status:404
```

## Step 5: Run the tunnel
```bash
cloudflared tunnel run lie-detector-viewer
```

## Step 6: Set up DNS
```bash
cloudflared tunnel route dns lie-detector-viewer lie-detector-viewer.your-domain.com
```

## Benefits:
- ✅ No port forwarding needed
- ✅ HTTPS automatically
- ✅ Works behind firewalls
- ✅ More secure than direct port exposure 