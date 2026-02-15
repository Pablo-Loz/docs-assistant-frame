"""
Bot Técnico Pipeline - Wrapper for Open WebUI.

Adds the pipelines directory to sys.path to enable package imports.
"""

import sys
from pathlib import Path

# Add pipelines directory to path for package imports
_pipelines_dir = Path(__file__).parent.resolve()
if str(_pipelines_dir) not in sys.path:
    sys.path.insert(0, str(_pipelines_dir))

# Now we can import the package properly
from bot_tecnico import Pipeline

# Re-export for Open WebUI discovery
__all__ = ["Pipeline"]
