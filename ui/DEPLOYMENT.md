# Data Viewer Deployment Guide

## Vercel Deployment

This Flask application is configured to deploy to Vercel with automatic CI/CD through GitHub Actions.

### Prerequisites

1. A Vercel account (https://vercel.com)
2. Vercel CLI installed locally: `npm i -g vercel`
3. GitHub repository with Actions enabled

### Initial Setup

1. **Link to Vercel Project** ✅ (Already completed)
   
2. **Get Vercel Credentials**
   - Go to https://vercel.com/account/tokens
   - Create a new token with full scope
   - Copy the token value

3. **Configure GitHub Secrets**
   Add these secrets to your GitHub repository (Settings → Secrets → Actions):
   - `VERCEL_TOKEN`: Your Vercel API token (create at https://vercel.com/account/tokens)
   - `VERCEL_ORG_ID`: `team_NmBfKxgMp9oJaNmoZtb0SNg4`
   - `VERCEL_PROJECT_ID`: `prj_C1gTgBtPk867euNkK62vvBNz4qaG`

4. **Production URL**
   Your app is deployed at: https://lies-j60tsu7hg-jackhopkins-projects.vercel.app

### Environment Variables

Configure these in Vercel dashboard (Project Settings → Environment Variables):
- `USE_LOCAL_DATA`: Set to `"False"` for production (enables S3 data loading)
- `S3_BUCKET`: `dipika-lie-detection-data` (already configured in vercel.json)
- `S3_PREFIX`: `processed-data/` (already configured in vercel.json)

**AWS Credentials** (Required for S3 access):
- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_DEFAULT_REGION`: `us-east-1` (or your bucket's region)

### Deployment Process

**Automatic Deployment:**
- Push to `main` or `feature/ui` branches triggers production deployment
- Pull requests trigger preview deployments with unique URLs

**Manual Deployment:**
```bash
cd data_viewer
vercel --prod
```

### CI/CD Workflow

The GitHub Actions workflow (`deploy-data-viewer.yml`) handles:
1. Running tests (when implemented)
2. Preview deployments for PRs
3. Production deployments for main/feature branches
4. Automatic PR comments with preview URLs

### Project Structure for Vercel

```
data_viewer/
├── api/
│   └── index.py       # Vercel entry point
├── app.py             # Flask application
├── templates/         # HTML templates
├── requirements.txt   # Python dependencies
└── vercel.json       # Vercel configuration
```

### Monitoring

- View deployments: https://vercel.com/[your-org]/[your-project]/deployments
- Check logs: https://vercel.com/[your-org]/[your-project]/functions

### Troubleshooting

1. **Import errors**: Ensure all imports in `app.py` are relative or absolute from project root
2. **Static files**: Place in `public/` directory or configure in `vercel.json`
3. **Environment variables**: Check they're set in Vercel dashboard for the correct environment
4. **Build failures**: Check GitHub Actions logs and Vercel build logs

### Local Testing

Test the Vercel build locally:
```bash
cd data_viewer
vercel dev
```

This will simulate the Vercel environment on your local machine.

## S3 Data Configuration

The application is configured to load data directly from S3 when deployed to production (`USE_LOCAL_DATA=False`).

### S3 Setup Requirements

1. **AWS IAM User** with S3 read permissions for the bucket `dipika-lie-detection-data`
2. **IAM Policy** should include at minimum:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "s3:GetObject",
           "s3:ListBucket"
         ],
         "Resource": [
           "arn:aws:s3:::dipika-lie-detection-data",
           "arn:aws:s3:::dipika-lie-detection-data/*"
         ]
       }
     ]
   }
   ```

### Data Structure Expected

The application expects JSON/JSONL files in S3 with the following structure:
```
s3://dipika-lie-detection-data/processed-data/
├── task-folder-1/
│   ├── file1.jsonl
│   └── file2.jsonl
├── task-folder-2/
│   ├── file1.jsonl
│   └── file2.jsonl
└── ...
```

Each JSONL file should contain one JSON object per line with fields like:
- `sample_id`, `task`, `task_id`, `timestamp`, `model`
- `trace`, `did_lie`, `evidence`, `metadata`, `scores`

### Performance Considerations

- The serverless function processes a maximum of 50 files per request to avoid timeouts
- Data is cached for 5 minutes to improve performance
- Use the `/refresh_data` endpoint to force reload data from S3

### Automatic S3 Sync

The application includes a **GitHub Actions workflow** that automatically syncs S3 data:

**Schedule**: Every 5 minutes (`sync-s3-data.yml`)
**Triggers**:
- Scheduled: Every 5 minutes via cron
- Manual: Workflow dispatch button in GitHub Actions
- Deployment: After successful deployments to main/feature branches

**How it works**:
1. Calls the `/refresh_data` endpoint on the production app
2. Forces fresh data load from S3
3. Updates the cached data for all users
4. Logs success/failure for monitoring

**Manual trigger**: Go to GitHub Actions → "Sync S3 Data" → "Run workflow"

### Local S3 Testing

To test S3 integration locally:
1. Set environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   export USE_LOCAL_DATA=False
   ```
2. Run the app: `python app.py`