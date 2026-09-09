"""Opt-in tool-selection smoke tests, with a real model and local knowledge."""

import os

import pytest

from arkb.config import DEFAULT_GENERATION_MODEL, RuntimeConfig
from arkb.knowledge.documents import DocumentAccess
from arkb.retrieval import BM25Retriever, RetrievalEngine
from arkb.runtime import Runtime


pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.environ.get('OBSIDIAN_RAG_RUN_MODEL_TESTS') != '1',
    reason='Set OBSIDIAN_RAG_RUN_MODEL_TESTS=1 with a tool-capable model cached and Ollama running.')]


@pytest.mark.parametrize('query,allowed_first_tools', [
    ('哪些笔记提到了 RAG', ('match',)),
    ('有哪些笔记和 RAG 相关', ('search',)),
    ('读取 rag.md', ('read',)),
    ('帮我找一些写 Agent Memory 的素材', ('match', 'search', 'read')),
    ('你好', ()),
])
def test_real_model_selects_tools_and_finishes(tmp_path, query, allowed_first_tools):
    (tmp_path / 'rag.md').write_text(
        '# RAG\nRAG uses retrieved knowledge to ground answers. RAG 结合检索和生成。', encoding='utf-8')
    (tmp_path / 'memory.md').write_text(
        '# Agent Memory\nAgent Memory stores past events and useful facts. '
        'Agent Memory 包括工作记忆与长期记忆。', encoding='utf-8')
    documents = DocumentAccess(tmp_path, vault_id='agent-test')
    # Keep the service check focused on model tool calling; search policy is
    # still host-configured and hidden from the model, with no vector server.
    engine = RetrievalEngine(bm25=BM25Retriever(list(documents.records()), index_id='test'))
    with Runtime(RuntimeConfig(timeout=120)) as runtime:
        tools = runtime.agent_tools(engine=engine, directory=tmp_path, vault_id='agent-test', mode='bm25')
        result = runtime.run_agent(query, tools=tools, max_turns=8,
                                   model=os.environ.get('OBSIDIAN_RAG_AGENT_MODEL', DEFAULT_GENERATION_MODEL))
    names = [c['function']['name'] for c in result.state.tool_calls]
    assert result.stop_reason == 'final', names
    assert result.response and result.response.strip()
    if not allowed_first_tools:
        assert names == []
    else:
        assert names and names[0] in allowed_first_tools
        assert all(name in {'match', 'search', 'read'} for name in names)
