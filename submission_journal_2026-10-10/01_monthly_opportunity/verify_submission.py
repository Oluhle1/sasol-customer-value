"""Validate this archived submission; this does not train a model."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1]/"tools"/"verify_package.py"),run_name="__main__")
