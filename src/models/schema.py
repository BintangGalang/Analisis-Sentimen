"""
src/models/schema.py
====================
Re-export skema data SocialMediaItem, Platform, DataType dari src.pipeline.schema
untuk kompatibilitas impor.
"""

from ..pipeline.schema import (
    Platform,
    DataType,
    EngagementMetrics,
    SocialMediaItem
)

__all__ = [
    "Platform",
    "DataType",
    "EngagementMetrics",
    "SocialMediaItem"
]

