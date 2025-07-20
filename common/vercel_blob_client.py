"""
Vercel Blob client for Python Flask app.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
import hashlib
from urllib.parse import quote


class VercelBlobClient:
    """Client for interacting with Vercel Blob via REST API."""

    def __init__(self):
        self.blob_read_write_token = os.environ.get('BLOB_READ_WRITE_TOKEN')
        self.enabled = bool(self.blob_read_write_token)

        if not self.enabled:
            print("Vercel Blob not configured. Blob operations will be disabled.")

    def put_blob(self, pathname: str, content: str, options: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Upload content as a blob with hierarchical pathname."""
        if not self.enabled:
            return None

        try:
            url = f"https://blob.vercel-storage.com/{quote(pathname, safe='/')}"

            headers = {
                "Authorization": f"Bearer {self.blob_read_write_token}",
                "x-content-type": "application/json"
            }

            # Add additional options as headers
            if options:
                for key, value in options.items():
                    headers[f"x-{key}"] = str(value)

            response = requests.put(url, data=content.encode('utf-8'), headers=headers)

            if response.status_code in [200, 201]:
                return response.json()
            else:
                print(f"Blob upload failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            print(f"Error uploading blob: {e}")
            return None

    def get_blob(self, url: str) -> Optional[str]:
        """Get blob content by URL."""
        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                return response.text
            else:
                return None

        except Exception as e:
            print(f"Error getting blob: {e}")
            return None

    def list_blobs(self, prefix: str = None, limit: int = 1000, cursor: str = None) -> Dict[str, Any]:
        """List blobs with optional prefix filtering."""
        if not self.enabled:
            return {"blobs": [], "hasMore": False, "cursor": None}

        try:
            url = "https://blob.vercel-storage.com/list"

            headers = {
                "Authorization": f"Bearer {self.blob_read_write_token}"
            }

            params = {"limit": limit}
            if prefix:
                params["prefix"] = prefix
            if cursor:
                params["cursor"] = cursor

            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Blob list failed: {response.status_code} - {response.text}")
                return {"blobs": [], "hasMore": False, "cursor": None}

        except Exception as e:
            print(f"Error listing blobs: {e}")
            return {"blobs": [], "hasMore": False, "cursor": None}

    def delete_blob(self, url: str) -> bool:
        """Delete a blob by URL."""
        if not self.enabled:
            return False

        try:
            delete_url = "https://blob.vercel-storage.com/delete"

            headers = {
                "Authorization": f"Bearer {self.blob_read_write_token}",
                "Content-Type": "application/json"
            }

            data = {"urls": [url]}

            response = requests.post(delete_url, headers=headers, json=data)

            return response.status_code == 200

        except Exception as e:
            print(f"Error deleting blob: {e}")
            return False

    def get_samples_by_filter(self, task: str = None, model: str = None, did_lie: bool = None) -> List[Dict[str, Any]]:
        """Get samples by filtering with hierarchical path structure: model/task/domain/sample.json"""
        if not self.enabled:
            return []

        try:
            # Build prefix based on filters - using hierarchical structure
            prefix_parts = []

            if model:
                prefix_parts.append(self._clean_name(model))
                if task:
                    prefix_parts.append(self._clean_name(task))
            elif task:
                # If no model specified, search across all models
                return self._search_across_models(task=task, did_lie=did_lie)

            prefix = "/".join(prefix_parts) if prefix_parts else ""

            # Get all blobs with prefix
            all_blobs = []
            cursor = None

            while True:
                result = self.list_blobs(prefix=prefix, cursor=cursor)
                blobs = result.get("blobs", [])
                all_blobs.extend(blobs)

                if not result.get("hasMore", False):
                    break
                cursor = result.get("cursor")

            # Filter and load samples
            samples = []
            for blob in all_blobs:
                if blob["pathname"].endswith(".json"):
                    content = self.get_blob(blob["url"])
                    if content:
                        try:
                            sample = json.loads(content)
                            # Apply did_lie filter if specified
                            if did_lie is None or sample.get("did_lie", False) == did_lie:
                                samples.append(sample)
                        except json.JSONDecodeError:
                            continue

            return samples

        except Exception as e:
            print(f"Error getting samples by filter: {e}")
            return []

    def _search_across_models(self, task: str = None, did_lie: bool = None) -> List[Dict[str, Any]]:
        """Search across all models for a specific task."""
        try:
            # Get all blobs and filter client-side
            all_blobs = []
            cursor = None

            while True:
                result = self.list_blobs(cursor=cursor)
                blobs = result.get("blobs", [])
                all_blobs.extend(blobs)

                if not result.get("hasMore", False):
                    break
                cursor = result.get("cursor")

            samples = []
            clean_task = self._clean_name(task) if task else None

            for blob in blobs:
                pathname = blob["pathname"]

                # Check if pathname matches our criteria: model/task/domain/sample.json
                if clean_task and clean_task not in pathname:
                    continue

                if pathname.endswith(".json"):
                    content = self.get_blob(blob["url"])
                    if content:
                        try:
                            sample = json.loads(content)
                            if did_lie is None or sample.get("did_lie", False) == did_lie:
                                samples.append(sample)
                        except json.JSONDecodeError:
                            continue

            return samples

        except Exception as e:
            print(f"Error searching across models: {e}")
            return []

    def get_unique_values(self) -> Dict[str, Any]:
        """Get unique values for filtering by analyzing blob paths."""
        if not self.enabled:
            return {}

        try:
            # Get all blobs
            all_blobs = []
            cursor = None

            while True:
                result = self.list_blobs(cursor=cursor)
                blobs = result.get("blobs", [])
                all_blobs.extend(blobs)

                if not result.get("hasMore", False):
                    break
                cursor = result.get("cursor")

            models = set()
            tasks = set()
            model_counts = {}
            task_counts = {}
            lie_counts = {"true": 0, "false": 0}

            for blob in all_blobs:
                pathname = blob["pathname"]
                if pathname.endswith(".json"):
                    # Parse path: model/task/domain/sample.json
                    parts = pathname.split("/")
                    if len(parts) >= 2:
                        model = parts[0]
                        task = parts[1]
                        models.add(model)
                        tasks.add(task)

                        model_counts[model] = model_counts.get(model, 0) + 1
                        task_counts[task] = task_counts.get(task, 0) + 1

                    # Get sample to check did_lie status
                    content = self.get_blob(blob["url"])
                    if content:
                        try:
                            sample = json.loads(content)
                            if sample.get("did_lie", False):
                                lie_counts["true"] += 1
                            else:
                                lie_counts["false"] += 1
                        except json.JSONDecodeError:
                            continue

            return {
                "tasks": [{"value": task, "count": task_counts.get(task, 0)} for task in sorted(tasks)],
                "models": [{"value": model, "count": model_counts.get(model, 0)} for model in sorted(models)],
                "lie_counts": lie_counts,
                "total_count": lie_counts["true"] + lie_counts["false"]
            }

        except Exception as e:
            print(f"Error getting unique values: {e}")
            return {}

    def _clean_name(self, name: str) -> str:
        """Clean name for use in blob paths."""
        return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace(":", "_")