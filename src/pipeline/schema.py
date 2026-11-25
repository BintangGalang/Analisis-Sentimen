"""
Modul Skema Data (Pydantic Models)
Menstandarisasi data post dan komentar dari semua platform media sosial.
"""

from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class Platform(str, Enum):
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    THREADS = "threads"
    TIKTOK = "tiktok"
    TWITTER = "twitter"
    YOUTUBE = "youtube"
    MAPS = "maps"


class DataType(str, Enum):
    POST = "post"
    COMMENT = "comment"


class EngagementMetrics(BaseModel):
    likes: int = 0
    comments_count: int = 0
    shares: int = 0
    views: int = 0


class SocialMediaItem(BaseModel):
    """
    Model representasi tunggal untuk setiap Post atau Komentar dari medsos.
    """
    id: str = Field(..., description="ID unik (misal: ig_post_123456 atau ig_comment_98765)")
    platform: Platform = Field(..., description="Media sosial asal (instagram, facebook, threads, tiktok)")
    data_type: DataType = Field(..., description="Tipe data: post atau comment")
    parent_id: Optional[str] = Field(None, description="ID Post utama jika ini adalah komentar")
    author: str = Field(..., description="Username atau nama pengguna")
    content: str = Field(..., description="Isi teks caption post atau komentar mentah")
    clean_content: Optional[str] = Field(None, description="Teks setelah proses pembersihan NLP")
    created_at: Optional[str] = Field(None, description="Waktu posting/komentar (ISO string atau teks relatif)")
    url: Optional[str] = Field(None, description="URL tautan langsung ke postingan/komentar")
    engagement: EngagementMetrics = Field(default_factory=EngagementMetrics)
    hashtag_query: str = Field(default="#undiksha", description="Kata kunci/hashtag pencarian")
    scraped_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp waktu data diambil")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata spesifik per platform jika ada")

    def to_flat_dict(self) -> Dict[str, Any]:
        """Konversi ke format datar (flat) agar mudah disimpan ke CSV / Pandas DataFrame."""
        return {
            "id": self.id,
            "platform": self.platform.value if isinstance(self.platform, Platform) else self.platform,
            "data_type": self.data_type.value if isinstance(self.data_type, DataType) else self.data_type,
            "parent_id": self.parent_id,
            "author": self.author,
            "content": self.content,
            "clean_content": self.clean_content,
            "created_at": self.created_at,
            "url": self.url,
            "likes": self.engagement.likes,
            "comments_count": self.engagement.comments_count,
            "shares": self.engagement.shares,
            "views": self.engagement.views,
            "hashtag_query": self.hashtag_query,
            "scraped_at": self.scraped_at,
        }
