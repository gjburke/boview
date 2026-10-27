"""Frame processing stages and end-to-end orchestration."""

from .frame import PipelineFrame
from .process import process_frame

__all__ = ["PipelineFrame", "process_frame"]
