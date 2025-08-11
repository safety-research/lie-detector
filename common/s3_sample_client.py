"""
S3 client for storing individual samples as JSON files.
Replaces the VercelBlobClient with direct S3 operations.
"""

import boto3
import json
import os
import re
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError


class S3SampleClient:
    """Client for storing individual samples as JSON files in S3."""
    
    def __init__(self, bucket: str = None, prefix: str = None):
        """
        Initialize S3 client.
        
        Args:
            bucket: S3 bucket name (defaults to env var S3_BUCKET)
            prefix: S3 prefix for samples (defaults to env var S3_PREFIX)
        """
        self.bucket = bucket or os.environ.get('S3_BUCKET', 'dipika-lie-detection-data')
        self.prefix = prefix or os.environ.get('S3_PREFIX', 'processed-data/')
        
        # Ensure prefix ends with /
        if self.prefix and not self.prefix.endswith('/'):
            self.prefix += '/'
        
        try:
            self.s3_client = boto3.client('s3')
            self.enabled = True
        except NoCredentialsError:
            print("AWS credentials not found. S3 operations will be disabled.")
            self.enabled = False
        except Exception as e:
            print(f"Error creating S3 client: {e}")
            self.enabled = False
    
    def generate_sample_id(self, original_sample_id: str) -> str:
        """
        Generate a clean sample ID from the original fully qualified name.
        
        Args:
            original_sample_id: Original sample ID like 'conv_20250715_sandbagging_pair_vandal_2_aeef35cba706_2025-07-15T22:44:04+01:00-SrFI8xmGhseU4xIcd9wSSTQ65mqc5G.json'
        
        Returns:
            Clean sample ID that preserves uniqueness
        """
        if not original_sample_id:
            return f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Remove .json extension if present
        sample_id = str(original_sample_id)
        if sample_id.endswith('.json'):
            sample_id = sample_id[:-5]
        
        # For inspect_ai generated IDs, try to extract meaningful parts while preserving uniqueness
        # Pattern: conv_20250717_sandbagging_pair_question_id_hash_timestamp-uniqueid
        parts = sample_id.split('_')
        
        if len(parts) >= 2:
            # Keep the base name and date, plus a unique identifier
            #base_part = f"{parts[0]}"#_{parts[1]}"  # e.g., "conv_20250717"
            
            # Look for a hash-like string or unique identifier in the original
            # Extract the last part after the last dash which is usually unique
            if '-' in sample_id:
                unique_suffix = sample_id.split('-')[-1]
                # Take first 8 characters of the unique suffix for brevity
                unique_suffix = unique_suffix[:8]
                return f"{unique_suffix}"
            
            # If no dash, use a hash of the full original ID to ensure uniqueness
            import hashlib
            hash_suffix = hashlib.md5(str(original_sample_id).encode()).hexdigest()[:8]
            return f"{hash_suffix}"
        
        # Fallback: use a hash of the original ID
        import hashlib
        hash_id = hashlib.md5(str(original_sample_id).encode()).hexdigest()[:12]
        return f"sample_{hash_id}"
    
    def put_sample(self, model: str, task: str, sample_id: str, content: Dict[str, Any]) -> bool:
        """
        Upload a sample as individual JSON file to S3.
        
        Args:
            model: Model name (e.g., "meta-llama/llama-3.1-8b-instruct")
            task: Task name (e.g., "sandbagging_physical_security_contrastive")
            sample_id: Sample identifier
            content: Sample content as dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Parse provider and model from model string
            provider, model_name = self._parse_model_name(model)
            
            # Parse task and domain from task string  
            task_type, domain = self._parse_task_name(task)
            
            # Clean names for S3 paths
            clean_provider = self._clean_name(provider)
            clean_model = self._clean_name(model_name)
            clean_task = self._clean_name(task_type)
            clean_domain = self._clean_name(domain)
            clean_sample_id = self.generate_sample_id(sample_id)

            did_lie = content['did_lie'] if 'did_lie' in content else "_"
            truth_tag = "t_" if did_lie else "f_"

            # Create S3 key: prefix/provider/model/domain/task/sample_id.json
            # This matches the structure from map_sample_to_s3_path
            key = f"{self.prefix}{clean_provider}/{clean_model}/{clean_domain}/{clean_task}/{truth_tag}{clean_sample_id}.json"
            
            # Convert content to JSON string
            json_content = json.dumps(content, indent=2)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json_content.encode('utf-8'),
                ContentType='application/json'
            )
            
            print(f"[S3SampleClient] Successfully uploaded sample to s3://{self.bucket}/{key}")
            return True
            
        except Exception as e:
            print(f"[S3SampleClient] Error uploading sample: {e}")
            return False
    
    def update_sample_with_baseline(self, model: str, task: str, sample_id: str, baseline_data: Dict[str, Any]) -> bool:
        """
        Update an existing sample file with baseline results using map_sample_to_s3_path.
        
        Args:
            model: Model name (e.g., "meta-llama/llama-3.1-8b-instruct")
            task: Task name (e.g., "sandbagging_physical_security_contrastive")
            sample_id: Sample identifier
            baseline_data: Baseline results to add
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Parse provider and model from model string
            provider, model_name = self._parse_model_name(model)
            
            # Parse task and domain from task string  
            domain, task_type = self._parse_task_name(task)
            
            # Clean names for S3 paths
            clean_provider = self._clean_name(provider)
            clean_model = self._clean_name(model_name)
            clean_domain = self._clean_name(domain)
            clean_task_type = self._clean_name(task_type)
            clean_sample_id = self.generate_sample_id(sample_id)

            # Extract did_lie from baseline_data to determine truth tag
            did_lie = baseline_data.get('did_lie', False)
            if isinstance(did_lie, str) and did_lie.lower() == 'n/a':
                did_lie = False
            elif isinstance(did_lie, str) and did_lie.lower() == 'true':
                did_lie = True
            elif isinstance(did_lie, str) and did_lie.lower() == 'false':
                did_lie = False
            truth_tag = "t_" if did_lie else "f_"

            # Create S3 key: prefix/provider/model/domain/task_type/sample_id.json
            key = f"{self.prefix}{clean_provider}/{clean_model}/{clean_domain}/{clean_task_type}/{truth_tag}{clean_sample_id}.json"
            
            # Use the configured bucket
            bucket_name = self.bucket
            
            # We now have bucket_name and key directly
            
            # Try to get existing sample data from the mapped path
            existing_data = {}
            try:
                response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
                existing_content = response['Body'].read().decode('utf-8')
                existing_data = json.loads(existing_content)
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    print(f"[S3SampleClient] Sample file not found at mapped path: {bucket_name}/{key}")
                    return False
                else:
                    raise e
            
            # Extract baseline type from baseline_data
            baseline_type = baseline_data.get('baseline_type', 'unknown')
            
            # Initialize baselines dict if it doesn't exist
            if 'baselines' not in existing_data:
                existing_data['baselines'] = {}
            
            # Add baseline results
            existing_data['baselines'][baseline_type] = {
                'prediction': baseline_data.get('prediction', 'OTHER'),
                'parseable': baseline_data.get('parseable', True),
                'accuracy': baseline_data.get('accuracy', 0.0)
            }
            
            # Convert updated content to JSON string
            json_content = json.dumps(existing_data, indent=2)
            
            # Upload updated content to S3 at the mapped path
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=json_content.encode('utf-8'),
                ContentType='application/json'
            )
            
            print(f"[S3SampleClient] Successfully updated sample with {baseline_type} baseline at s3://{bucket_name}/{key}")
            return True
            
        except Exception as e:
            print(f"[S3SampleClient] Error updating sample with baseline: {e}")
            return False
    
    def get_sample(self, model: str, task: str, sample_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a sample from S3.
        
        Args:
            model: Model name
            task: Task name
            sample_id: Sample identifier
            
        Returns:
            Sample content as dictionary or None if not found
        """
        if not self.enabled:
            return None
        
        try:
            # Clean names for S3 paths
            clean_model = self._clean_name(model)
            clean_task = self._clean_name(task)
            clean_sample_id = self.generate_sample_id(sample_id)
            
            # Create S3 key
            key = f"{self.prefix}{clean_model}/{clean_task}/{clean_sample_id}.json"
            
            # Get from S3
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            
            return json.loads(content)
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            print(f"[S3SampleClient] Error getting sample: {e}")
            return None
        except Exception as e:
            print(f"[S3SampleClient] Error getting sample: {e}")
            return None
    
    def list_samples(self, model: str = None, task: str = None) -> List[str]:
        """
        List sample keys in S3.
        
        Args:
            model: Optional model filter
            task: Optional task filter
            
        Returns:
            List of S3 keys
        """
        if not self.enabled:
            return []
        
        try:
            # Build prefix based on filters
            prefix_parts = [self.prefix]
            
            if model:
                prefix_parts.append(f"{self._clean_name(model)}/")
                if task:
                    prefix_parts.append(f"{self._clean_name(task)}/")
            
            search_prefix = "".join(prefix_parts)
            
            # List objects in S3
            paginator = self.s3_client.get_paginator('list_objects_v2')
            keys = []
            
            for page in paginator.paginate(Bucket=self.bucket, Prefix=search_prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('.json'):
                        keys.append(key)
            
            return keys
            
        except Exception as e:
            print(f"[S3SampleClient] Error listing samples: {e}")
            return []
    
    def delete_sample(self, model: str, task: str, sample_id: str) -> bool:
        """
        Delete a sample from S3.
        
        Args:
            model: Model name
            task: Task name
            sample_id: Sample identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Clean names for S3 paths
            clean_model = self._clean_name(model)
            clean_task = self._clean_name(task)
            clean_sample_id = self.generate_sample_id(sample_id)
            
            # Create S3 key
            key = f"{self.prefix}{clean_model}/{clean_task}/{clean_sample_id}.json"
            
            # Delete from S3
            self.s3_client.delete_object(Bucket=self.bucket, Key=key)
            
            print(f"[S3SampleClient] Successfully deleted sample from s3://{self.bucket}/{key}")
            return True
            
        except Exception as e:
            print(f"[S3SampleClient] Error deleting sample: {e}")
            return False
    
    def _parse_model_name(self, model: str) -> Tuple[str, str]:
        """
        Parse model name into provider and model.
        
        Args:
            model: Model name like "meta-llama/llama-3.1-8b-instruct"
            
        Returns:
            Tuple of (provider, model_name)
        """
        if "/" in model:
            if "openrouter" in model:
                model = model.replace("openrouter/", "")
            parts = model.split("/", 1)
            provider = parts[0].replace("-", "_")
            model_name = parts[1].replace("-", "_").replace(".", "_")
            return provider, model_name
        else:
            # If no slash, treat entire string as model name with unknown provider
            return "unknown", model.replace("-", "_").replace(".", "_")
    
    def _parse_task_name(self, task: str) -> Tuple[str, str]:
        """
        Parse task name into domain and task type.
        
        Args:
            task: Task name like "ascii_train" or "mask_continuations"
            
        Returns:
            Tuple of (domain, task_type)
        """
        # For task names like "ascii_train", "mask_continuations", "sandbagging_train"
        # We want domain first, then task_type
        
        parts = task.split("_")
        if len(parts) >= 3:
            # For longer names like "sandbagging_physical_security_contrastive"
            # Assume first part is domain, rest is task_type
            domain = parts[0]
            task_type = "_".join(parts[1:])
            return domain, task_type
        elif len(parts) == 2:
            # Two parts: domain and task_type
            # e.g., "ascii_train" -> ("ascii", "train")
            return parts[0], parts[1]
        else:
            # Single part: use as domain with "general" as task_type
            return task, "general"
    
    def _clean_name(self, name: str) -> str:
        """Clean name for use in S3 paths."""
        return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace(":", "_")