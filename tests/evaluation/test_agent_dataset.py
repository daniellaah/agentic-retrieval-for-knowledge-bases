from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from arkb.evaluation.datasets import load_agent_eval_dataset, parse_agent_eval_dataset
from arkb.evaluation.models import AgentEvalCase


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / 'evaluation/data/agent_v1.jsonl'
NOTES = ROOT / 'example_notes'


def test_versioned_dataset_loads_in_order_with_real_sources_and_no_runtime_fields():
    cases = load_agent_eval_dataset(DATASET, notes_dir=NOTES)
    assert Counter(c.task_type for c in cases) == {
        'exact_lookup': 6, 'semantic_discovery': 6, 'direct_read': 6,
        'exploratory_retrieval': 10, 'knowledge_qa': 6, 'no_retrieval': 6,
    }
    assert [c.id for c in cases] == [json.loads(line)['id'] for line in DATASET.read_text().splitlines()]
    assert all(c.notes for c in cases)
    assert not {'trace', 'response', 'expected_tools'} & asdict(cases[0]).keys()
    assert parse_agent_eval_dataset(
        b'\n'.join(json.dumps(asdict(c)).encode() for c in cases), notes_dir=NOTES,
    ) == cases


def row(**overrides):
    return {'id': 'one', 'query': 'Read a.md', 'task_type': 'direct_read',
            'expected_sources': ['a.md'], **overrides}


@pytest.mark.parametrize('overrides, message', [
    ({'id': ''}, 'id'), ({'id': 1}, 'id'), ({'query': ' '}, 'query'),
    ({'task_type': 'unknown'}, 'task_type'), ({'task_type': []}, 'task_type'),
    ({'expected_sources': 'a.md'}, 'expected_sources'),
    ({'expected_sources': [None]}, 'expected_sources'),
    ({'expected_sources': ['a.md', 'a.md']}, 'duplicates'),
    ({'expected_sources': ['../a.md']}, 'filenames'),
    ({'expected_sources': ['/a.md']}, 'filenames'),
    ({'expected_sources': ['a.txt']}, 'filenames'),
    ({'expected_sources': ['a.md ']}, 'whitespace'),
    ({'expected_sources': []}, 'nonempty'),
    ({'task_type': 'no_retrieval'}, 'no_retrieval'),
    ({'forbidden_tools': ['browse']}, 'unknown tool'),
    ({'allowed_tools': 'read'}, 'array'),
    ({'allowed_tools': ['read'], 'forbidden_tools': ['read']}, 'overlap'),
    ({'max_tool_calls': True}, 'max_tool_calls'), ({'max_tool_calls': -1}, 'max_tool_calls'),
    ({'min_source_recall': float('nan')}, 'min_source_recall'),
    ({'min_source_recall': True}, 'min_source_recall'),
    ({'min_source_recall': .5}, 'full source recall'),
    ({'min_read_sources': 2}, 'min_read_sources'),
    ({'min_read_sources': True}, 'min_read_sources'),
    ({'notes': None}, 'notes'), ({'expected_tools': ['read']}, 'unexpected'),
])
def test_invalid_schema_fails_with_line_context(overrides, message):
    with pytest.raises(ValueError, match=f'line 2:.*{message}'):
        parse_agent_eval_dataset(b'\n' + json.dumps(row(**overrides)).encode())


@pytest.mark.parametrize('raw, message', [
    (b'{', 'line 1'), (b'[]', 'JSON object'), (b'{}', 'required'),
    (b'\n ', 'at least one'), (b'{"id":"one","id":"two"}', 'duplicate JSON field'),
])
def test_malformed_jsonl(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_agent_eval_dataset(raw)


def test_duplicate_id_is_rejected():
    raw = json.dumps(row()).encode()
    with pytest.raises(ValueError, match='line 3:.*duplicate case id'):
        parse_agent_eval_dataset(raw + b'\n\n' + raw)


def test_existence_validation_is_optional_and_respects_document_scope(tmp_path):
    raw = json.dumps(row()).encode()
    assert parse_agent_eval_dataset(raw)[0].expected_sources == ('a.md',)
    with pytest.raises(ValueError, match='source does not exist'):
        parse_agent_eval_dataset(raw, notes_dir=tmp_path)
    (tmp_path / 'a.md').write_text('# A\nEvidence')
    assert parse_agent_eval_dataset(raw, notes_dir=tmp_path)[0].id == 'one'
    directory = tmp_path / 'notes'
    directory.mkdir()
    (directory / 'a.md').symlink_to(tmp_path / 'a.md')
    with pytest.raises(ValueError, match='source does not exist'):
        parse_agent_eval_dataset(raw, notes_dir=directory)
    with pytest.raises(ValueError, match='not a directory'):
        parse_agent_eval_dataset(raw, notes_dir=tmp_path / 'missing')


def test_annotations_are_detached_from_mutable_input():
    values = row()
    case = AgentEvalCase(**values)
    values['expected_sources'].append('b.md')
    assert case.expected_sources == ('a.md',)
