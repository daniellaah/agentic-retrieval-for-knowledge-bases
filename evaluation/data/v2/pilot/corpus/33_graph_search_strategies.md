# GraphRAG Search: Global Summaries, Local Exploration, and Deferred Work

A corpus can support both narrow factual questions and broad questions about recurring themes. Those tasks may need different retrieval paths. The Microsoft Research articles summarized here describe several GraphRAG approaches and the tradeoffs between preparing summaries in advance and selecting evidence during a query. They do not establish a universal best architecture.

## Summarize communities for global questions

The original GraphRAG approach extracts entities and relationships from source material, organizes the resulting graph into communities, and prepares community summaries. These summaries provide a view across connected passages. A question about the main themes of an entire collection may be poorly served by only a few passages that resemble its wording. Global search can instead form partial responses from community reports and combine them into an answer. This adds work at indexing time and introduces derived representations that need grounding in the underlying corpus. A community report is useful for synthesis, but it is not interchangeable with an original passage when a claim requires exact supporting detail. The design targets a different information need from a single isolated fact lookup. This section summarizes source 1.

## Prune irrelevant branches dynamically

Dynamic community selection changes which community reports participate in global search. Instead of processing every report at one fixed hierarchy level, the system assesses relevance while traversing the hierarchy. It starts with higher-level communities, discards irrelevant branches, and descends into relevant children. Selected reports can therefore come from different levels of abstraction. A relatively lightweight relevance decision can avoid sending many unsuitable reports through the more expensive answer-generation stage. This selection policy trades additional routing decisions for less unnecessary downstream work. Its success depends on preserving branches that contain useful evidence; early rejection can hide an entire region of the collection. Evaluating only the final synthesis would make that routing failure difficult to diagnose. This section summarizes source 2.

## Use community context to guide local exploration

DRIFT combines global context with local search. Its Primer examines community information and develops an initial response together with follow-up questions. The Follow-Up phase explores those questions using local retrieval, progressively adding detail. The resulting Output Hierarchy records a structure of questions and answers that can support the final response. This is more than choosing between a global mode and a local mode at the start. Broader context helps decide which local investigations are worth pursuing. The stages serve different roles: community-level material helps orient the investigation, while local evidence fills in specifics. Extra exploration has a cost, so question generation and stopping behavior affect both the coverage and expense of the process. This section summarizes source 3.

## Defer expensive reasoning until a query arrives

LazyGraphRAG explores a different balance between indexing and query-time computation. It builds a graph using noun phrases and their co-occurrence, with community structure, without requiring the same LLM-generated summaries during indexing. Query-time exploration then performs relevance testing and selective reasoning over the collection. Its search combines broad exploration with deeper investigation of promising areas. A relevance test budget controls how much assessment is allowed, providing a way to vary effort and quality. Moving computation later can make ingestion cheaper, but does not make answering free. The tradeoff depends on corpus update frequency, query volume, and how much query-specific investigation is required. This is a research design whose reported results should be interpreted within its evaluated setting. This section summarizes source 4.

## Illustrative scenario: a collection of project retrospectives

Consider an imaginary archive of project retrospectives. One question asks which recurring causes delayed delivery across the archive. Another asks which dependency blocked a particular project. This scenario is original and contains no actual company records. The first question calls for coverage across projects; the second needs precise evidence from a smaller connected neighborhood.

A global summary could help identify themes such as unclear ownership or dependency changes, but a reviewer would still want source passages supporting any specific project claim. If dynamic selection excludes a relevant community too early, the synthesis may appear coherent while missing a substantial theme. Recording selected communities would help distinguish that error from poor answer writing.

A DRIFT-style investigation might begin with the recurring themes, then ask focused follow-up questions about exceptions. A lazy approach would spend less effort generating reports before anyone asks a question, while allocating more work when the question arrives. A useful experiment would compare the same questions and evidence requirements across these choices. Measuring only the number of retrieved documents would overlook whether the search actually covered the requested parts of the archive.

## Sources

Accessed 2026-09-05. Sections 1–4 correspond to the sources below; the scenario is original.

1. [Microsoft Research — GraphRAG: new tool for complex data discovery](https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/)
2. [Microsoft Research — Dynamic community selection](https://www.microsoft.com/en-us/research/blog/graphrag-improving-global-search-via-dynamic-community-selection/)
3. [Microsoft Research — Introducing DRIFT Search](https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/)
4. [Microsoft Research — LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
