# AWS Setup for S3 Access

This application requires AWS credentials to access the S3 bucket `dipika-lie-detection-data`.

## Option 1: AWS CLI Configuration (Recommended)

1. Install AWS CLI if not already installed:
   ```bash
   pip install awscli
   ```

2. Configure AWS credentials:
   ```bash
   aws configure
   ```
   
   Enter your:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region (e.g., us-east-1)
   - Default output format (json)

## Option 2: Environment Variables

Set the following environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_access_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_access_key
export AWS_DEFAULT_REGION=us-east-1
```

## Option 3: IAM Role (for EC2 instances)

If running on an EC2 instance, you can attach an IAM role with S3 read permissions for the bucket `dipika-lie-detection-data`.

## Required S3 Permissions

The application needs the following S3 permissions:
- `s3:ListBucket` on `dipika-lie-detection-data`
- `s3:GetObject` on `dipika-lie-detection-data/generated-data/*`

## Testing S3 Access

You can test your S3 access with:

```bash
aws s3 ls s3://dipika-lie-detection-data/generated-data/
```

This should list the folders in your generated-data directory. 