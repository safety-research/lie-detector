#!/bin/bash

echo "🚀 Deploying Data Viewer to Vercel"
echo "================================="

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Please install it with: npm i -g vercel"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Please run this script from the data_viewer directory"
    exit 1
fi

# First time setup
if [ ! -d ".vercel" ]; then
    echo "📦 First time setup - linking to Vercel project..."
    vercel link
fi

# Deploy to production
echo "🔄 Deploying to production..."
vercel --prod

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Add these secrets to GitHub (Settings → Secrets → Actions):"
echo "   - VERCEL_TOKEN (from https://vercel.com/account/tokens)"
echo "   - VERCEL_ORG_ID (from .vercel/project.json)"
echo "   - VERCEL_PROJECT_ID (from .vercel/project.json)"
echo ""
echo "2. Push changes to trigger automatic deployment:"
echo "   git add -A"
echo "   git commit -m 'Add Vercel deployment for data_viewer'"
echo "   git push"