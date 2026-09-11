# Document Ingestion: Structure, Provenance, and Retrieval Units

Document ingestion determines what information a retrieval system can later find and cite. Converting a file into plain text, deciding retrieval boundaries, and retaining source metadata are separate decisions. This note follows those decisions from layout extraction to index preparation. The scenario at the end is an original design exercise.

## Recover a document's hierarchy

Google Document AI's Layout Parser represents document structure rather than treating the input as an undifferentiated character stream. Headings, paragraphs, lists, and tables provide boundaries and relationships that can guide chunk construction. Ancestor headings can supply context when a small passage would otherwise lose its place in the document. Table headers matter because a cell value without its row or column meaning may be misleading. Depending on the input and parser capabilities, visual descriptions can also contribute information. Layout parsing and chunk construction remain conceptually distinct: identifying a section is an extraction result, while deciding how much of that section to retrieve is a downstream policy. A structurally plausible output still deserves inspection against the original document. This section summarizes source 1.

## Carry location through the pipeline

Google Cloud demonstrates a pipeline that calls Document AI through BigQuery's ML.PROCESS_DOCUMENT function. The returned document structure can be transformed into chunk records before embeddings are generated. Alongside the text, the pipeline retains metadata such as the source URI, page location, and structural information. These fields help filter results and diagnose why a particular passage entered an answer. A source pointer should survive later transformations, rather than exist only in an initial upload record. The blog also discusses incremental processing as new source documents arrive. The broader lesson is that indexing consists of several linked transformations: retrieval text alone is insufficient for tracing a result back through parsing to the original file. This section summarizes source 2.

## Inspect parsing and manage duplicates

Databricks' RAG data-pipeline guidance recommends examining parsed output and preserving useful metadata. Metadata can describe the document, its content, its internal structure, or its ingestion context. Examples include document identifiers and versions, subjects, section or page information, and processing timestamps. These categories support different diagnostic questions and should not be collapsed into a single vague label. Duplicate material can occupy multiple retrieval positions without adding independent evidence. Deduplication may use known metadata or similarity techniques such as MinHash, depending on the corpus. Selecting a retained version should consider provenance, authority, and freshness rather than arbitrary input order. Improving downstream ranking cannot reconstruct a table or paragraph that parsing already omitted. This section summarizes source 3.

## Build chunks from meaningful elements

Unstructured distinguishes partitioning a document into elements from combining those elements into retrieval chunks. A heading, paragraph, and short list item need not each become a separate vector. Small elements can be grouped, while oversized elements may require splitting. A title-aware strategy can respect section boundaries; character separators alone have less understanding of document structure. Very small chunks can lose definitions and conditions, while large chunks can mix unrelated topics and dilute the representation. A maximum token allowance is a hard constraint, not evidence that the largest permitted chunk is optimal. Useful boundaries preserve enough nearby context for a fragment to make sense independently. Chunk size therefore belongs in an evaluation with realistic questions. This section summarizes source 4.

## Illustrative scenario: a revised operations manual

Imagine an internal operations manual with a troubleshooting table and two editions. This is an invented example. A parser extracts the numeric entries but accidentally omits the column headings. A search system can still retrieve those numbers, yet the generator cannot reliably tell whether they represent a timeout, retry limit, or version number. The first investigation should inspect extraction quality before adjusting retrieval parameters.

Now suppose the older and newer editions both describe the same procedure. The ingestion record keeps the document identity, edition, section heading, page, and extracted text. A reviewer can compare a suspicious answer with the correct source edition. If duplicate passages occupy both retrieved positions, the team can examine version selection separately from semantic similarity.

For a future chunking experiment, a question might require a definition and the exception in the following paragraph. Keeping the exception nearby could matter more than reaching an arbitrary word count. The evaluation should record that evidence relationship explicitly. It should also include questions whose answers appear in tables, so that successful prose extraction does not conceal a broken table path.

## Sources

Accessed 2026-09-05. Sections 1–4 correspond to the sources below; the scenario is original.

1. [Google Cloud documentation — Layout parsing and chunking](https://docs.cloud.google.com/document-ai/docs/layout-parse-chunk)
2. [Google Cloud — Building RAG pipelines with BigQuery and Layout Parser](https://cloud.google.com/blog/products/data-analytics/bigquery-and-document-ai-layout-parser-for-document-preprocessing)
3. [Databricks documentation — Improve RAG data pipeline quality](https://docs.databricks.com/aws/en/agents/tutorials/ai-cookbook/quality-data-pipeline-rag)
4. [Unstructured — Chunking for RAG: best practices](https://unstructured.io/blog/chunking-for-rag-best-practices)
