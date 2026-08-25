"""Shared API clients and the probed embedding dimension.

Separate from `config` so that importing configuration does not require API keys.
Anything that needs to talk to a provider imports from here; anything that only
needs a setting does not.
"""

import os

from openai import OpenAI
from pinecone import Pinecone

from .config import EMBED_MODEL

client = OpenAI()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Probed rather than assumed. Dimension varies with the model and, for
# Matryoshka-capable models, with the requested `dimensions` parameter — so a
# hard-coded 1536 becomes wrong the moment anyone changes EMBED_MODEL.
EMBED_DIMS = len(
    client.embeddings.create(model=EMBED_MODEL, input=["probe"]).data[0].embedding
)
