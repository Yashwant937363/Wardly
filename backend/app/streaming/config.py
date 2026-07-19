import os
from typing import Final

# streaming/config.py

# Stream names — the actual Redis keys for each stage's queue
STREAM_CLASSIFY = "stream:classify"
STREAM_SEGMENT  = "stream:segment"
STREAM_ANALYZE  = "stream:analyze"
STREAM_EMBED    = "stream:embed"
STREAM_SAVE     = "stream:save"

# Consumer group names — one group per stage
GROUP_CLASSIFY = "classify-workers"
GROUP_SEGMENT  = "segment-workers"
GROUP_ANALYZE  = "analyze-workers"
GROUP_EMBED    = "embed-workers"
GROUP_SAVE     = "save-workers"

# Redis connection settings
REDIS_HOST = os.getenv("REDIS_HOST","localhost")
REDIS_PORT: Final[int] = int(os.getenv("REDIS_PORT", 6379))

# Optional: how long a message can be pending before it's considered "stuck"
CLAIM_IDLE_TIME_MS = 30_000  # 30 seconds