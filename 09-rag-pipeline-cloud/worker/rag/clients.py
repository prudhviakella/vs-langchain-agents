"""Shared API clients, and the probed embedding dimension.

    import rag.clients
            |
            |-- create the OpenAI client
            |-- create the Pinecone client
            |
            v
    embed one throwaway string
            |
            v
    len(vector)  ->  EMBED_DIMS


WHY THE DIMENSION IS PROBED, NOT WRITTEN DOWN
---------------------------------------------

A hard-coded 1536 is correct until someone changes EMBED_MODEL, and then it
is silently wrong:

    creating an index with the wrong dimension fails loudly, which is fine
    QUERYING one returns plausible scores and no error, which is not

Costing one tiny API call at import to get this right is a good trade.


WHY THIS IS SEPARATE FROM config.py
-----------------------------------

Importing configuration should not require API keys.

Anything that only needs a setting imports config. Anything that needs to
talk to a provider imports this. That split is why six of the ten modules in
this package can be imported with no credentials at all.
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
