# Query Planning: Conversation Rewriting, Candidate Recall, and Context Compression

A user's message is not always a useful search query, and a retrieved document is not always the best form of context for answering. This note follows transformations before and after candidate retrieval. It distinguishes changing the search request from changing the evidence supplied to the generator.

## Resolve a follow-up into a search question

LangChain describes query transformations that adapt conversational requests for retrieval. A follow-up may contain pronouns or refer to something named in an earlier turn. A standalone search question can restore the missing subject while preserving the user's intended constraints. Sending the entire conversation to a retriever may introduce unrelated topics, so rewriting requires selecting relevant context. Other transformations include generating multiple query variants or asking a broader step-back question. These approaches change what the retriever searches for. They should be evaluated for whether they preserve the actual information need, because a well-formed rewritten question can still refer to the wrong entity or omit an important restriction. This section summarizes source 1.

## Match the query form to the operation

Elastic discusses rewriting natural-language requests into forms better suited to search, including enriched terms and queries influenced by a hypothetical answer. A request that needs filtering, counting, or aggregation can require a structured operation rather than only a semantic similarity lookup. Expansion should retain the original constraints; additional terms can broaden discovery without replacing what the user actually asked. A hypothetical answer is a search aid and should not be treated as verified source evidence. That distinction matters because generation used during query preparation can introduce plausible but unsupported details. The value of rewriting depends on the retrieval task and the candidate collection, so it needs measurement rather than an assumption that more query text always helps. This section summarizes source 2.

## Improve candidate coverage before reranking

Microsoft's discussion of query rewriting and semantic ranking places rewriting before first-stage retrieval. That initial stage can use lexical, vector, or hybrid search to collect candidates. A later semantic ranker evaluates the shortlist more closely; a cross-encoder judges the query and candidate together rather than relying only on independently stored embeddings. These stages address complementary problems. Query rewriting can help useful evidence enter the candidate set, while reranking can improve its ordering once present. Reranking cannot recover a passage that never entered its shortlist. Its more expensive joint scoring is therefore applied to a limited set. An evaluation should inspect candidate coverage as well as final ordering so that missing evidence is not misdiagnosed as a ranking-only problem. This section summarizes source 3.

## Compress retrieved context with the question in view

LangChain's contextual compression approach operates after retrieval. A query-aware compressor can extract relevant portions of a document and discard documents that do not help answer the question. This reduces the material passed to the generator while preserving useful evidence where the compressor succeeds. It differs from summarizing a conversation history: the input is retrieved material, and the current question guides what to retain. It also differs from adding explanatory context before indexing. Compression cannot recover evidence absent from the retrieved set, and an overly aggressive extraction can remove a qualification needed for a faithful answer. Both document selection and retained passage content therefore deserve inspection. This section summarizes source 4.

## Illustrative scenario: a follow-up about a revised service

Imagine a fictional conversation comparing two versions of a service. After several turns about unrelated deployment details, the user asks whether it supports the earlier exception. This is an original example. A query rewriter needs to resolve which service version and exception are intended. Copying the entire conversation into the query could emphasize the unrelated deployment discussion, while a careless short rewrite could choose the wrong version.

Suppose the corrected query retrieves a manual containing the relevant rule and its qualification. A reranker can move that manual upward if it appears in the candidate set. A context compressor then has to retain the qualification, not merely the sentence that sounds like a direct answer. If the qualification is omitted, the generator can produce a confident but incomplete response despite successful document retrieval.

The evaluation records the rewritten query, candidate set, final ordering, and retained evidence separately. A missing manual points toward candidate generation. A useful manual ranked too low points toward ordering. A lost exception points toward context selection. These observations suggest different experiments and prevent a single final-answer score from concealing where information was lost. They also provide a basis for later comparing whole-document retrieval with independently retrieved chunks.

## Sources

Accessed 2026-09-05. Sections 1–4 correspond to the sources below; the scenario is original.

1. [LangChain — Query transformations](https://www.langchain.com/blog/query-transformations)
2. [Elastic — Query rewriting to improve search](https://www.elastic.co/search-labs/blog/query-rewriting-llm-search-improve)
3. [Microsoft — Query rewriting and semantic ranking for RAG](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/raising-the-bar-for-rag-excellence-query-rewriting-and-new-semantic-ranker/4302729)
4. [LangChain — Contextual compression](https://www.langchain.com/blog/improving-document-retrieval-with-contextual-compression)
