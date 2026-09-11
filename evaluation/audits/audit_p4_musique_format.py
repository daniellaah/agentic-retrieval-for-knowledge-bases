"""Posthoc output-format sensitivity using one unambiguous JSON code block.

No model calls, field repair, answer changes or selection by gold correctness.
The frozen strict benchmark predictions remain unchanged.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from arkb.evaluation.external import (
    ExternalDataset, digest, read_jsonl, verify_checksums, write_json,
)
from arkb.evaluation.multihop import musique_metrics, parse_prediction


def extract_single_json_block(text):
    blocks = re.findall(r'^```([^\n]*)\n(.*?)^```[ \t]*$', text,
                        flags=re.MULTILINE | re.DOTALL)
    if len(blocks) != 1 or blocks[0][0].strip().lower() not in ('json', ''):
        return None
    return blocks[0][1].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify_checksums(args.run)
    if json.loads((args.run / 'experiment.json').read_text())['status'] != 'completed':
        raise ValueError('Require the full terminal run.')
    rows = read_jsonl(args.run / 'rows.jsonl')
    cases = json.loads((args.run / 'selected.json').read_text())['rows']
    original = read_jsonl(args.run / 'predictions.jsonl')
    if len(rows) != len(cases) or len(rows) != len(original):
        raise ValueError('Incomplete paired input.')
    predictions, decisions = [], []
    for i, (row, case, strict) in enumerate(zip(rows, cases, original)):
        if row['case_index'] != i or row['id'] != case['id'] or strict != row['prediction']:
            raise ValueError('Prediction identity differs.')
        prediction = strict
        action = 'kept_strict_valid' if row['parse_error'] is None else 'kept_nonfinal'
        error = row['parse_error']
        if row['stop_reason'] == 'final' and row['parse_error']:
            # Extraction uses only the final output. Source validation uses the
            # original public corpus mapping, never answers/support labels.
            block = extract_single_json_block(row['result']['response'])
            action = 'no_unambiguous_json_block'
            if block is not None:
                corpus = [{'id':str(p['idx']), 'title':p['title'], 'text':p['paragraph_text']}
                          for p in case['paragraphs']]
                data = ExternalDataset(f'musique-{i}', corpus, {'q':case['question']}, {'q':{}}, {}, {})
                sources = {s:int(idx) for s,idx in data.source_map().items()}
                parsed, error = parse_prediction(block, sources, stopped='final')
                action = 'block_schema_invalid'
                if error is None:
                    prediction = {'id':case['id'], **parsed}
                    action = 'extracted_unique_valid_block'
        predictions.append(prediction)
        decisions.append({'case_index':i, 'id':case['id'], 'action':action,
                          'strict_error':row['parse_error'], 'remaining_error':error})
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'predictions.jsonl').open('w') as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    write_json(args.output / 'decisions.json', decisions)
    write_json(args.output / 'summary.json', {
        'schema':'arkb-p4-posthoc-format-sensitivity-v1',
        'status':'posthoc_diagnostic_not_primary_score',
        'policy':'Keep all strict-valid outputs and nonfinal failures. For a strict-invalid final only, extract exactly one json/unlabelled fenced code block, then apply the original exact schema/source checks. No field repair or choice by gold.',
        'counts':dict(Counter(d['action'] for d in decisions)),
        'strict_metrics':musique_metrics(cases, original),
        'diagnostic_metrics':musique_metrics(cases, predictions),
        'answerability_confusion':{
            str(gold).lower():dict(Counter('undefined' if p['predicted_answerable'] is None
                else str(p['predicted_answerable']).lower()
                for c,p in zip(cases,predictions) if c['answerable'] is gold))
            for gold in (True,False)},
        'rows_sha256':digest(args.run / 'rows.jsonl'),
        'strict_predictions_sha256':digest(args.run / 'predictions.jsonl'),
        'script_sha256':digest(Path(__file__)),
        'limits':'A diagnostic on already exposed outputs; not a new model run, predeclared benchmark result, causal model improvement or reviewed semantic quality.',
        'release_eligible':False,
    })
    write_json(args.output / 'manifest.json', {'files':{
        p.name:digest(p) for p in args.output.iterdir() if p.is_file()}})
    print((args.output / 'summary.json').read_text())


if __name__ == '__main__':
    main()
