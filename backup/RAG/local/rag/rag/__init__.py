"""Document ingestion into a vector index.

    from rag import config, docling_io, chunking, sync

Modules, in the order the pipeline uses them:

    config       every setting both halves must agree on
    clients      API clients and the probed embedding dimension
    docling_io   parsing a PDF, and reading the objects Docling returns
    inspect      verifying the parse, and the readable reports
    tables       table serialisation, structure checks, summaries
    chunking     records, metadata, and table summary chunks
    embedding    embedding with a content-addressed cache
    index        index access and the manifest
    sync         the three-way diff that makes re-ingestion cheap
    retrieval    searching the index and answering from what comes back
    audit        the DynamoDB trail, when running on AWS

The notebooks call these in order. The same package runs on AWS unchanged.
"""
