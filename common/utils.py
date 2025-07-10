import boto3
from typing import Optional

def write_to_s3(data: str, filename: str, bucket: str = "processed-", prefix: str = "processed-data/", clean_task_name: Optional[str] = None):
    """
    Write data to S3 bucket.
    
    Args:
        data: String data to write
        filename: Name of the file to create in S3
        bucket: S3 bucket name
        prefix: S3 key prefix
        clean_task_name: Clean task name (with - instead of _) to use as subdirectory
    """
    try:
        s3_client = boto3.client('s3')
        
        # Create the S3 key with optional subdirectory
        if clean_task_name:
            key = f"{prefix}{clean_task_name}/{filename}"
        else:
            key = f"{prefix}{filename}"
        
        # Upload the data
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data.encode('utf-8'),
            ContentType='application/json'
        )
        
        print(f"[S3Upload] Successfully uploaded to s3://{bucket}/{key}")
        return True
    except Exception as e:
        print(f"[S3Upload] Error uploading to S3: {e}")
        return False