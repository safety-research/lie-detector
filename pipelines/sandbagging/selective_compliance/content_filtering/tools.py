"""
Tools for the tool results sandbagging detection module.

Implements a virtual filesystem with file reading capabilities.
"""

from inspect_ai.tool import tool
from typing import Dict, Optional
import json


class VirtualFilesystem:
    """A virtual filesystem containing files with various content."""
    
    def __init__(self):
        self.files: Dict[str, str] = {}
        self.read_history: Dict[str, int] = {}
    
    def add_file(self, path: str, content: str):
        """Add a file to the virtual filesystem."""
        self.files[path] = content
        self.read_history[path] = 0
    
    def read_file(self, path: str) -> Optional[str]:
        """Read a file from the virtual filesystem."""
        if path in self.files:
            self.read_history[path] += 1
            return self.files[path]
        return None
    
    def list_files(self) -> list[str]:
        """List all files in the virtual filesystem."""
        return sorted(list(self.files.keys()))
    
    def get_read_history(self) -> Dict[str, int]:
        """Get the history of file reads."""
        return self.read_history.copy()
    
    def was_file_read(self, path: str) -> bool:
        """Check if a file was read."""
        return self.read_history.get(path, 0) > 0


# Global filesystem instance for the current evaluation
_filesystem: Optional[VirtualFilesystem] = None


def init_filesystem(files: Dict[str, str]) -> VirtualFilesystem:
    """Initialize the virtual filesystem with files."""
    global _filesystem
    _filesystem = VirtualFilesystem()
    for path, content in files.items():
        _filesystem.add_file(path, content)
    return _filesystem


def get_filesystem() -> Optional[VirtualFilesystem]:
    """Get the current filesystem instance."""
    return _filesystem


@tool
def read_file():
    async def execute(path: str) -> str:
        """
        Read a file from the virtual filesystem.

        Args:
            path: The path to the file to read

        Returns:
            The content of the file if it exists, or an error message
        """
        fs = get_filesystem()
        if fs is None:
            return "Error: Filesystem not initialized"
        
        content = fs.read_file(path)
        if content is None:
            available_files = fs.list_files()
            return f"Error: File '{path}' not found. Available files: {', '.join(available_files)}"
        
        return content
    
    return execute


@tool
def list_files():
    async def execute() -> str:
        """
        List all files in the virtual filesystem.

        Returns:
            A list of file paths
        """
        fs = get_filesystem()
        if fs is None:
            return "Error: Filesystem not initialized"
        
        files = fs.list_files()
        if not files:
            return "No files found in the filesystem."
        
        return "Files in the filesystem:\n" + "\n".join(f"- {f}" for f in files)
    
    return execute


@tool 
def get_file_info():
    async def execute(path: str) -> str:
        """
        Get information about a file including its size and type.

        Args:
            path: The path to the file

        Returns:
            Information about the file
        """

        fs = get_filesystem()
        if fs is None:
            return "Error: Filesystem not initialized"
        
        content = fs.files.get(path)
        if content is None:
            return f"Error: File '{path}' not found"
        
        # Determine file type based on extension
        if path.endswith('.txt'):
            file_type = "Text file"
        elif path.endswith('.md'):
            file_type = "Markdown file"
        elif path.endswith('.json'):
            file_type = "JSON file"
        elif path.endswith('.csv'):
            file_type = "CSV file"
        elif path.endswith('.log'):
            file_type = "Log file"
        else:
            file_type = "Unknown file type"
        
        info = {
            "path": path,
            "type": file_type,
            "size": len(content),
            "size_human": f"{len(content)} bytes",
            "lines": content.count('\n') + 1 if content else 0
        }
        
        return json.dumps(info, indent=2)
    
    return execute