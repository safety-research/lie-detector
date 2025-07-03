# Simple Setup for Continuous Running

## Option 1: Systemd Service (Recommended)

1. **Install Gunicorn** (for production):
```bash
cd data_viewer
conda activate lie-detector
pip install gunicorn
```

2. **Copy the service file** to systemd:
```bash
sudo cp lie-detector-viewer.service /etc/systemd/system/
```

3. **Enable and start the service**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable lie-detector-viewer
sudo systemctl start lie-detector-viewer
```

4. **Check status**:
```bash
sudo systemctl status lie-detector-viewer
```

5. **View logs**:
```bash
sudo journalctl -u lie-detector-viewer -f
```

## Option 2: Simple Background Process

If you don't want to use systemd, you can use `nohup`:

```bash
cd data_viewer
conda activate lie-detector
nohup python app.py > app.log 2>&1 &
```

This will keep it running even after you disconnect from SSH.

## Access

Once running, the app will be available at:
- **Local**: http://localhost:9009
- **External**: http://your-server-ip:9009 (if firewall allows)

## Stop the Service

**For systemd**:
```bash
sudo systemctl stop lie-detector-viewer
```

**For nohup**:
```bash
# Find the process
ps aux | grep "python app.py"
# Kill it
kill <process_id>
``` 