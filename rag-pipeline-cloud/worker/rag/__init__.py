"""Document ingestion into a vector index.

    PDF
     |
     v
   docling_io.py     layout, TableFormer, OCR, classifier, CodeFormula
     |               then describe_figures() — vision model, CACHED
     |
     v
   inspect.py        did it work? counts, layout quality, a readable report
     |
     v
   headings.py       correct false headings before chunking
     |
     v
   tables.py         serialise, check the grid, summarise
     |
     v
   chunking.py       drop furniture, split, merge prose, one record per figure
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

WHAT THE CHUNKER DOES NOT DO FOR YOU

HybridChunker only splits, and merges consecutive chunks with an equal heading
path. It never filters and never enforces a minimum size. Three policies are
therefore ours, and they live in chunking.py:

    what is not worth indexing      the drop filter
    how small a prose chunk may be  the merge
    one record per figure           the figure pass

WHAT THE LAYOUT MODEL DECIDES

Which regions are headings, and which captions belong to which figure. Both
are wrong often enough to matter, neither has a pipeline flag, and the only
lever is LAYOUT_MODEL in docling_io.py. The extraction report prints both
numbers, so comparing two layout models is reading two lines.

The notebooks call these in order. The same package runs on AWS unchanged.
"""
