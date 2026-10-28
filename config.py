"""Shared settings for documentation ingestion and question retrieval."""

from pathlib import Path


EMBEDDING_MODEL = "text-embedding-3-small"
DATA_DIR = Path(__file__).parent / "data"
GUIDES = {
    "user-guide": "https://docs.aws.amazon.com/bedrock/latest/userguide/",
    "api-reference": "https://docs.aws.amazon.com/bedrock/latest/APIReference/",
}
