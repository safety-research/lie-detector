# ngrok Setup (Quick External Access)

## Step 1: Install ngrok
```bash
# Download ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin/
```

## Step 2: Sign up and get authtoken
1. Go to https://ngrok.com/signup
2. Get your authtoken from the dashboard
3. Run: `ngrok config add-authtoken YOUR_TOKEN`

## Step 3: Expose your Flask app
```bash
ngrok http 9009
```

## Step 4: Access from anywhere
ngrok will give you a public URL like:
```
https://abc123.ngrok.io
```

Anyone can access your app using this URL!

## Benefits:
- ✅ No configuration needed
- ✅ Works immediately
- ✅ HTTPS included
- ✅ Free tier available 