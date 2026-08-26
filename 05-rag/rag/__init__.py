"""Document ingestion into a vector index.

    PDF
     |
     v
   docling_io.py     six models -> a DoclingDocument
     |
     v
   inspect.py        did it work? counts, warnings, a readable report
     |
     v
   tables.py         serialise, check the grid, summarise
     |
     v
   chunking.py       split, add heading context, one summary per table
     |
     v
   embedding.py      records -> vectors, cached by content
     |
     v
   sync.py           what changed? added / removed / moved / unchanged
     |
     v
   index.py          Pinecone, and the manifest both halves must agree on

   retrieval.py      the other direction: question -> passages -> answer

Supporting:
   config.py         every setting both halves must agree on
   clients.py        API clients, and the probed embedding dimension
   audit.py          the DynamoDB trail, when running on AWS

THE IDEA THAT SHAPES ALL OF IT

When this pipeline goes wrong it usually does not crash. A setting left off
means an equation becomes a placeholder, a chart never gets described, or a
table comes out with a broken grid — and every step after that runs perfectly
happily on top of it.

You end up with an index that looks complete and is missing content. Nothing
downstream can detect it.

So each stage has a checkpoint, and `inspect.py` writes reports you read next
to the original PDF before anything is embedded.

The notebooks call these in order. The same package runs on AWS unchanged.
"""
