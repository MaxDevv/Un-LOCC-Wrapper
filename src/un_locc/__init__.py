"""
Un-LOCC: Universal Lossy Optical Context Compression

A wrapper library for the OpenAI SDK that adds optical compression capabilities
to text inputs in chat completions and responses APIs.
"""

from .un_locc import UnLOCC, AsyncUnLOCC

__version__ = "0.1.0"
__all__ = ["UnLOCC", "AsyncUnLOCC"]
