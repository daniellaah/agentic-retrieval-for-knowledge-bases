"""Versioned evidence annotations and deterministic v2 scores; no inference.

Provisional annotations are usable only by explicit opt-in and never imply
independent human review. Coordinates reference Note.content, not raw Markdown.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from arkb.evaluation.datasets import _unique_object
from arkb.knowledge.documents import load_notes

SCHEMA = 'arkb-eval-v2'
TASKS = {'semantic_discovery', 'exploratory_retrieval', 'knowledge_qa', 'multi_hop_qa',
         'exact_lookup', 'direct_read', 'evidence_gap', 'no_retrieval'}


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _fields(row, required, optional=()):
    if not isinstance(row, dict) or set(row) - set(required) - set(optional) or set(required) - set(row):
        raise ValueError(f'Invalid fields; expected {sorted(required)}, optional {sorted(optional)}.')


def _strings(values, *, nonempty=False):
    return (isinstance(values, list) and (bool(values) or not nonempty)
            and all(_text(v) for v in values) and len(values) == len(set(values)))


def _status(row):
    if row['annotation_status'] not in ('provisional', 'reviewed') or not _strings(row['reviewers']):
        raise ValueError('Invalid annotation status/reviewers.')
    if row['annotation_status'] == 'reviewed' and not row['reviewers']:
        raise ValueError('Reviewed labels require identified reviewers.')


def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON: {value}')))


def _read_rows(path):
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line, object_pairs_hook=_unique_object,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON: {value}'))))
        except ValueError as error:
            raise ValueError(f'{path.name}:{number}: {error}') from error
    return rows


@dataclass(frozen=True)
class EvalDataset:
    manifest: dict
    cases: tuple[dict, ...]
    evidence: dict[str, dict]
    qrels: tuple[dict, ...]
    bodies: dict[str, str]

    @property
    def provisional(self):
        return any(row['annotation_status'] != 'reviewed'
                   for row in (*self.cases, *self.evidence.values(), *self.qrels))


def load_dataset(directory: Path, *, notes_dir: Path, allow_provisional=False) -> EvalDataset:
    directory, notes_dir = Path(directory), Path(notes_dir)
    manifest = _read_json(directory / 'manifest.json')
    _fields(manifest, {'schema_version','corpus_id','files','corpus','description'})
    if manifest['schema_version'] != SCHEMA or not _text(manifest['corpus_id']):
        raise ValueError('Unsupported dataset schema/corpus.')
    if not isinstance(manifest['files'],dict) or set(manifest['files']) != {'queries.jsonl','evidence.jsonl','qrels.jsonl'}:
        raise ValueError('Manifest must hash all three data files.')
    for name, digest in manifest['files'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Dataset checksum mismatch: {name}.')
    notes = {note.source: note for note in load_notes(notes_dir)}
    if not isinstance(manifest['corpus'],list) or not manifest['corpus']:
        raise ValueError('Corpus manifest must be nonempty.')
    recorded = set()
    for doc in manifest['corpus']:
        _fields(doc, {'source','document_revision','body_sha256','origin'})
        source = doc['source']
        if not _text(source) or source in recorded or source not in notes or not (notes_dir/source).resolve().is_relative_to(notes_dir.resolve()):
            raise ValueError('Invalid, duplicate, or external corpus source.')
        recorded.add(source)
        note = notes[source]
        if note.document_revision != doc['document_revision'] or hashlib.sha256(note.content.encode()).hexdigest() != doc['body_sha256']:
            raise ValueError(f'Corpus revision mismatch: {source}.')
    if recorded != set(notes):
        raise ValueError('Corpus file set differs from manifest.')
    evidence = {}
    for span in _read_rows(directory/'evidence.jsonl'):
        _fields(span, {'id','source','document_revision','start_char','end_char','quote','quote_sha256','annotation_status','reviewers'})
        _status(span)
        if not _text(span['id']) or span['id'] in evidence or span['source'] not in notes:
            raise ValueError('Invalid/duplicate evidence ID or source.')
        note = notes[span['source']]
        start,end = span['start_char'],span['end_char']
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(note.content):
            raise ValueError('Invalid evidence body coordinates.')
        if (span['document_revision'] != note.document_revision or note.content[start:end] != span['quote']
                or hashlib.sha256(span['quote'].encode()).hexdigest() != span['quote_sha256']):
            raise ValueError('Evidence revision/quote mismatch.')
        evidence[span['id']] = span
    cases, ids, families, queries = [], set(), {}, {}
    for case in _read_rows(directory/'queries.jsonl'):
        _fields(case, {'id','intent_family_id','task_type','query','query_language','split','answerability',
                      'query_origin','annotation_status','reviewers','evidence_requirements','required_facts','expected_behavior'},
                {'exact_pattern','read_source'})
        _status(case)
        if any(not _text(case[k]) for k in ('id','intent_family_id','query','query_language','query_origin','expected_behavior')):
            raise ValueError('Missing case text/identity.')
        if case['id'] in ids or case['task_type'] not in TASKS or case['split'] not in ('dev','validation','test'):
            raise ValueError('Invalid/duplicate case identity, task, or split.')
        if case['answerability'] not in ('answerable','partial','unanswerable','needs_clarification','not_applicable'):
            raise ValueError('Invalid answerability.')
        if case['task_type']=='no_retrieval' and case['answerability']!='not_applicable':
            raise ValueError('no_retrieval requires not_applicable answerability.')
        if case['task_type'] not in ('no_retrieval','evidence_gap') and case['answerability']!='answerable':
            raise ValueError('Use evidence_gap for partial/unanswerable/ambiguous requests.')
        if case['task_type']=='evidence_gap' and case['answerability'] not in ('partial','unanswerable','needs_clarification'):
            raise ValueError('evidence_gap must describe a gap or ambiguity.')
        family = case['intent_family_id']
        if family in families and families[family] != case['split']:
            raise ValueError('Intent family leaks across splits.')
        families[family]=case['split']
        query_key=' '.join(case['query'].casefold().split())
        if query_key in queries:
            raise ValueError('Duplicate normalized query.')
        queries[query_key]=case['id']
        if not _strings(case['required_facts']):
            raise ValueError('Invalid required facts.')
        if not isinstance(case['evidence_requirements'],list):
            raise ValueError('Evidence requirements must be an array.')
        facet_ids=set()
        for facet in case['evidence_requirements']:
            _fields(facet, {'id','weight','critical','alternatives'})
            if (not _text(facet['id']) or facet['id'] in facet_ids or type(facet['critical']) is not bool
                    or type(facet['weight']) not in (int,float) or not math.isfinite(facet['weight']) or facet['weight']<=0):
                raise ValueError('Invalid facet identity/weight/critical flag.')
            facet_ids.add(facet['id'])
            if not isinstance(facet['alternatives'],list) or not facet['alternatives']:
                raise ValueError('Facet requires nonempty alternatives.')
            alternatives=set()
            for alternative in facet['alternatives']:
                if not _strings(alternative,nonempty=True) or set(alternative)-evidence.keys():
                    raise ValueError('Invalid/unknown evidence in alternative.')
                key=tuple(sorted(alternative))
                if key in alternatives:
                    raise ValueError('Duplicate evidence alternative.')
                alternatives.add(key)
        if case['task_type']=='no_retrieval' and case['evidence_requirements']:
            raise ValueError('no_retrieval must not require knowledge evidence.')
        if (case['answerability'] in ('answerable','partial') and not case['evidence_requirements']
                and case['task_type']!='exact_lookup'):
            raise ValueError('Answerable retrieval requires evidence.')
        if case['task_type']=='exact_lookup':
            _fields(case.get('exact_pattern'), {'query','regex','case_sensitive','target'})
            pattern=case['exact_pattern']
            if (not _text(pattern['query']) or type(pattern['regex']) is not bool
                    or type(pattern['case_sensitive']) is not bool or pattern['target'] not in ('content','source')):
                raise ValueError('Invalid exact pattern.')
        elif 'exact_pattern' in case:
            raise ValueError('Only exact_lookup has an exact pattern.')
        if case['task_type']=='direct_read':
            if case.get('read_source') not in notes:
                raise ValueError('Invalid direct read source.')
        elif 'read_source' in case:
            raise ValueError('Only direct_read has a read source.')
        ids.add(case['id']); cases.append(case)
    if not cases:
        raise ValueError('Dataset must not be empty.')
    qrels, pairs = [], set()
    for rel in _read_rows(directory/'qrels.jsonl'):
        _fields(rel, {'query_id','source','grade','annotation_status','reviewers'})
        _status(rel)
        key=(rel['query_id'],rel['source'])
        if key in pairs or key[0] not in ids or key[1] not in notes or type(rel['grade']) is not int or not 0<=rel['grade']<=3:
            raise ValueError('Invalid/duplicate relevance judgment.')
        pairs.add(key); qrels.append(rel)
    qrel_map={(r['query_id'],r['source']):r['grade'] for r in qrels}
    for case in cases:
        required={span for facet in case['evidence_requirements'] for alt in facet['alternatives'] for span in alt}
        if any(qrel_map.get((case['id'],evidence[e]['source']),0)<2 for e in required):
            raise ValueError('Required evidence lacks a supporting document judgment.')
        if case['task_type']=='no_retrieval' and any(q==case['id'] and grade>0 for (q,s),grade in qrel_map.items()):
            raise ValueError('no_retrieval cannot have positive knowledge judgments.')
    dataset=EvalDataset(manifest,tuple(cases),evidence,tuple(qrels),{s:n.content for s,n in notes.items()})
    if dataset.provisional and not allow_provisional:
        raise ValueError('Provisional dataset requires explicit allow_provisional=True; not a release gold set.')
    return dataset


def ranking_scores(qrels: dict[str,int], ranked: list[str], *, k=10, rel_threshold=2):
    """Explicit cutoff, exponential gain, zero-padded precision/judgment depth."""
    from arkb.evaluation.metrics import ranking_metrics
    # Reuse v1 validation and its exponential DCG without changing v1 semantics.
    graded=ranking_metrics(qrels,ranked,k=k)
    if any(grade>3 for grade in qrels.values()) or type(rel_threshold) is not int or not 1<=rel_threshold<=3:
        raise ValueError('v2 relevance grades are 0..3 and threshold is 1..3.')
    top=ranked[:k]
    relevant={s for s,g in qrels.items() if g>=rel_threshold}
    return {f'recall@{k}':len(relevant.intersection(top))/len(relevant) if relevant else None,
            f'mrr@{k}':next((1/i for i,s in enumerate(top,1) if s in relevant),0.) if relevant else None,
            f'ndcg_exp@{k}':graded['ndcg_at_k'],
            f'precision@{k}':len(relevant.intersection(top))/k if relevant else None,
            f'assessed@{k}':sum(s in qrels for s in top)/k,
            'returned_count':len(top),'unassessed_returned_count':sum(s not in qrels for s in top)}


def observation_intervals(observations: list[dict], dataset: EvalDataset):
    """Validate observed verbatim bodies/ranges and retain per-source intervals."""
    intervals={}
    rejected=0
    revisions={s['source']:s['document_revision'] for s in dataset.manifest['corpus']}
    for hit in observations:
        if not isinstance(hit,dict):
            rejected+=1; continue
        source=hit.get('source'); start,end=hit.get('start_char'),hit.get('end_char')
        # Filename match can return a complete body without range metadata.
        # Derive its range only on exact full-body equality, never by substring.
        if (isinstance(source,str) and source in dataset.bodies and start is None and end is None
                and hit.get('content')==dataset.bodies[source]):
            start,end=0,len(dataset.bodies[source])
        if (not isinstance(source,str) or source not in dataset.bodies
                or hit.get('document_revision')!=revisions[source]
                or type(start) is not int or type(end) is not int
                or not 0<=start<end<=len(dataset.bodies[source])
                or hit.get('content')!=dataset.bodies[source][start:end]):
            rejected+=1; continue
        intervals.setdefault(source,[]).append((start,end))
    return intervals,rejected


def unique_evidence_tokens(observations,dataset,*,counter):
    """Tokenize the union of valid returned intervals once per source/revision.

    Disjoint intervals are tokenized separately; this is an offline context
    redundancy diagnostic, not the online cumulative delivery allowance.
    """
    intervals,rejected=observation_intervals(observations,dataset)
    tokens=spans=0
    for source,ranges in intervals.items():
        merged=[]
        for start,end in sorted(ranges):
            if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
            else:merged.append([start,end])
        for start,end in merged:
            value=counter(dataset.bodies[source][start:end])
            if type(value) is not int or value<0:raise ValueError('Invalid reference token count.')
            tokens+=value;spans+=1
    return {'union_reference_tokens':tokens,'merged_intervals':spans,'documents':len(intervals),'rejected_observations':rejected}


def evidence_scores(case: dict, observations: list[dict], dataset: EvalDataset):
    """Only exact observed bodies/slices at the labeled revision cover spans.

    Accepts actual observations, not arguments, citations or hypothetical chunks.
    """
    intervals,rejected=observation_intervals(observations,dataset)
    fractions={}
    for facet in case['evidence_requirements']:
        for alt in facet['alternatives']:
            for eid in alt:
                span=dataset.evidence[eid]
                start,end=span['start_char'],span['end_char']
                cursor,covered=start,0
                for left,right in sorted(intervals.get(span['source'],[])):
                    left,right=max(start,left),min(end,right)
                    if left<right:
                        covered+=max(0,right-max(cursor,left)); cursor=max(cursor,right)
                fractions[eid]=covered/(end-start)
    satisfied={f['id']:any(all(fractions[e]==1 for e in alt) for alt in f['alternatives'])
               for f in case['evidence_requirements']}
    weight=sum(f['weight'] for f in case['evidence_requirements'])
    critical=[satisfied[f['id']] for f in case['evidence_requirements'] if f['critical']]
    return {'evidence_coverage':sum(f['weight']*satisfied[f['id']] for f in case['evidence_requirements'])/weight if weight else None,
            'all_critical_facets_covered':all(critical) if critical else None,
            'facet_satisfied':satisfied,'span_coverage':fractions,'rejected_observations':rejected,
            'annotation_status':'provisional' if dataset.provisional else 'reviewed',
            'answer_correct':None, 'grounded_task_success':None}


def dataset_report(dataset):
    return {'schema_version':SCHEMA,'case_count':len(dataset.cases),
            'intent_families':len({c['intent_family_id'] for c in dataset.cases}),
            'by_task_type':dict(Counter(c['task_type'] for c in dataset.cases)),
            'by_split':dict(Counter(c['split'] for c in dataset.cases)),
            'evidence_count':len(dataset.evidence),'qrel_count':len(dataset.qrels),
            'corpus_documents':len(dataset.bodies),'provisional':dataset.provisional,
            'release_eligible':False if dataset.provisional else None}


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path);p.add_argument('--notes-dir',type=Path,required=True)
    p.add_argument('--allow-provisional',action='store_true')
    a=p.parse_args(argv)
    try:
        dataset=load_dataset(a.dataset,notes_dir=a.notes_dir,allow_provisional=a.allow_provisional)
    except (ValueError,TypeError,KeyError,OSError) as error:
        p.error(str(error))
    print(json.dumps(dataset_report(dataset),ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
