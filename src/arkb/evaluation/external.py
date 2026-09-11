"""External corpus adapters and native-unit metrics, independent of v2 gold semantics.

Labels live outside the materialized knowledge directory. No remote code is
executed; optional Parquet/trec_eval dependencies are imported only when used.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import zipfile

from arkb.knowledge.models import Note, ChunkRecord
from arkb.knowledge.chunking import whole_note_chunks


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    import os
    import tempfile
    path=Path(path)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as f:
        temporary=Path(f.name)
        try:
            f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True);raise
    temporary.replace(path)


def verify_checksums(directory):
    directory=Path(directory).resolve()
    checks=json.loads((directory/'checksums.json').read_text())
    for name,expected in checks.items():
        path=directory/name
        if not path.resolve().is_relative_to(directory) or path.is_symlink() or digest(path)!=expected:
            raise ValueError('Run checksum mismatch: '+name)
    return len(checks)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def opaque_source(dataset, document_id):
    return hashlib.sha256(json.dumps([dataset,document_id], ensure_ascii=False).encode()).hexdigest()+'.md'


def sample_ids(ids, count, *, seed=20260910):
    """Order-independent sampling with a persisted ID list; no label inspection."""
    ids=list(ids)
    if len(ids)!=len(set(ids)) or type(count) is not int or count<1 or count>len(ids):
        raise ValueError('Invalid unique sample population/count.')
    return sorted(ids, key=lambda x: hashlib.sha256(f'{seed}:{x}'.encode()).digest())[:count]


@dataclass
class ExternalDataset:
    name: str
    corpus: list[dict]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    aspects: dict[str, list[dict]]
    provenance: dict

    def validate(self):
        ids=[d['id'] for d in self.corpus]
        if not ids or any(not isinstance(x,str) or not x for x in ids) or len(ids)!=len(set(ids)):
            raise ValueError('Corpus IDs must be nonempty and unique.')
        if any(set(d)!={'id','title','text'} or not isinstance(d['title'],str)
               or not isinstance(d['text'],str) for d in self.corpus):
            raise ValueError('Only document ID, title and text belong in corpus.')
        if not self.queries or any(not isinstance(q,str) or not q or not isinstance(t,str) or not t.strip()
                                   for q,t in self.queries.items()):
            raise ValueError('Invalid queries.')
        if set(self.qrels)!=set(self.queries) or not set(self.aspects)<=set(self.queries):
            raise ValueError('Query/label coverage mismatch.')
        known=set(ids)
        for qid,rels in self.qrels.items():
            if not set(rels)<=known or any(type(v) is not int or v<0 for v in rels.values()):
                raise ValueError('Invalid qrels or missing corpus positives.')
            seen=set(); aids=set()
            for aspect in self.aspects.get(qid,[]):
                docs=aspect['supporting_docs']; weight=aspect['weight']
                if (aspect['id'] in aids or type(weight) not in (int,float) or not math.isfinite(weight)
                        or weight<=0 or not docs or len(docs)!=len(set(docs)) or seen.intersection(docs)
                        or any(rels.get(d,0)<=0 for d in docs)):
                    raise ValueError('Aspects require distinct positive documents and finite positive weights.')
                aids.add(aspect['id']); seen.update(docs)
            if self.aspects.get(qid) and seen!={d for d,r in rels.items() if r>0}:
                raise ValueError('Aspect annotations must cover every positive.')
        return self

    def notes(self):
        return [Note(title=' '.join(d['title'].splitlines()).strip(), content=d['text'].strip(),
                     source=opaque_source(self.name,d['id'])) for d in self.corpus]

    def records(self):
        notes=self.notes()
        return [ChunkRecord.from_note(c, note=n, vault_id=self.name)
                for n,c in zip(notes,whole_note_chunks(notes))]

    def source_map(self):
        return {opaque_source(self.name,d['id']):d['id'] for d in self.corpus}

    def save(self, output, *, materialize=False):
        self.validate(); output=Path(output); output.mkdir(parents=True,exist_ok=False)
        with (output/'corpus.jsonl').open('w') as f:
            for d in self.corpus: f.write(json.dumps(d,ensure_ascii=False)+'\n')
        write_json(output/'queries.json',self.queries); write_json(output/'qrels.json',self.qrels)
        write_json(output/'aspects.json',self.aspects)
        if materialize: self.materialize(output/'corpus')
        write_json(output/'manifest.json', {'schema':'arkb-external-v1','name':self.name,
            'corpus_count':len(self.corpus),'query_count':len(self.queries),'provenance':self.provenance,
            'query_ids':list(self.queries),'normalization':'title lines joined; body stripped; raw corpus preserved',
            'ranking_unit':'original corpus ID','exposure':'public-development','release_gold':False,
            'files':{n:digest(output/n) for n in ('corpus.jsonl','queries.json','qrels.json','aspects.json')}})

    def materialize(self, directory):
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
        for n in self.notes():
            (directory/n.source).write_text('# '+n.title+'\n'+n.content,encoding='utf-8')

    def verify_materialized(self, directory):
        directory=Path(directory); notes=self.notes()
        paths=list(directory.iterdir())
        if {p.name for p in paths}!={n.source for n in notes} or any(p.is_symlink() for p in paths):
            raise ValueError('Materialized corpus file set changed.')
        for n in notes:
            if (directory/n.source).read_text(encoding='utf-8')!='# '+n.title+'\n'+n.content:
                raise ValueError('Materialized corpus text changed: '+n.source)


def load_external(directory):
    directory=Path(directory); m=json.loads((directory/'manifest.json').read_text())
    expected={'corpus.jsonl','queries.json','qrels.json','aspects.json'}
    if m['schema']!='arkb-external-v1' or set(m['files'])!=expected:
        raise ValueError('Invalid external manifest.')
    if any(digest(directory/n)!=h for n,h in m['files'].items()):
        raise ValueError('External input checksum mismatch.')
    data=ExternalDataset(m['name'],read_jsonl(directory/'corpus.jsonl'),
        *[json.loads((directory/n).read_text()) for n in ('queries.json','qrels.json','aspects.json')],m['provenance']).validate()
    if len(data.corpus)!=m['corpus_count'] or list(data.queries)!=m['query_ids'] or len(data.queries)!=m['query_count']:
        raise ValueError('External manifest counts differ.')
    return data


def _parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def import_public(inputs, name, *, query_limit=None):
    inputs=Path(inputs); downloads=json.loads((inputs/'downloads.json').read_text())
    prefixes={'scifact':('scifact/','scifact-qrels/'),
              'bright-stackoverflow':('bright-pro/',), 'bright-robotics':('bright-pro/',)}
    if name not in prefixes: raise ValueError('Unknown external dataset.')
    files={n:v for n,v in downloads['files'].items() if n.startswith(prefixes[name])}
    if name=='scifact':
        required={'scifact/README.md','scifact/corpus/corpus-00000-of-00001.parquet',
                  'scifact/queries/queries-00000-of-00001.parquet','scifact-qrels/test.tsv'}
    else:
        domain=name.removeprefix('bright-')
        required={'bright-pro/README.md','bright-pro/LICENSE',
                  *[f'bright-pro/{kind}/{domain}.parquet' for kind in ('documents','examples','aspects')]}
    if not required<=set(files):raise ValueError('Required download hashes are missing.')
    for n,v in files.items():
        if digest(inputs/n)!=v['sha256']: raise ValueError('Downloaded file changed: '+n)
    aspects={}
    if name=='scifact':
        docs=_parquet(inputs/'scifact/corpus/corpus-00000-of-00001.parquet')
        queries={r['_id']:r['text'] for r in _parquet(inputs/'scifact/queries/queries-00000-of-00001.parquet')}
        with (inputs/'scifact-qrels/test.tsv').open() as f: judgments=list(csv.DictReader(f,delimiter='\t'))
        split='test'; scope='full BEIR corpus'
    else:
        domain=name.removeprefix('bright-')
        docs=[{'_id':r['id'],'title':'','text':r['content']} for r in _parquet(inputs/f'bright-pro/documents/{domain}.parquet')]
        examples=_parquet(inputs/f'bright-pro/examples/{domain}.parquet')
        queries={str(r['id']):r['query'] for r in examples}
        judgments=[{'query-id':str(r['id']),'corpus-id':d,'score':1} for r in examples for d in r['gold_ids']]
        for a in _parquet(inputs/f'bright-pro/aspects/{domain}.parquet'):
            qid=a['id'].removeprefix(domain+'-').rsplit('-a',1)[0]
            aspects.setdefault(qid,[]).append(a)
        split=domain; scope='full per-domain Bright-Pro corpus'
    qrels=defaultdict(dict)
    for r in judgments:
        q,d,v=str(r['query-id']),str(r['corpus-id']),int(r['score'])
        if d in qrels[q]: raise ValueError('Duplicate qrel.')
        qrels[q][d]=v
    selected=sorted(qrels)
    if query_limit is not None: selected=sample_ids(selected,query_limit)
    corpus=[{'id':str(d['_id']),'title':d.get('title') or '', 'text':d['text']} for d in docs]
    return ExternalDataset(name,corpus,{q:queries[q] for q in selected},{q:qrels[q] for q in selected},
        {q:aspects[q] for q in selected if q in aspects},
        {'split':split,'scope':scope,'files':files,'query_limit':query_limit,'sample_seed':20260910}).validate()


def musique_groups(inputs, *, groups=100):
    """Keep the supplied context per variant; never combine contexts across queries."""
    inputs=Path(inputs); manifest=json.loads((inputs/'downloads.json').read_text())
    if digest(inputs/'musique.zip')!=manifest['files']['musique.zip']['sha256']:
        raise ValueError('MuSiQue archive checksum mismatch.')
    with zipfile.ZipFile(inputs/'musique.zip') as z:
        rows=[json.loads(line) for line in z.open('data/musique_full_v1.0_dev.jsonl')]
    by_id=defaultdict(list)
    for r in rows: by_id[r['id']].append(r)
    pairs={q:rs for q,rs in by_id.items() if len(rs)==2 and {r['answerable'] for r in rs}=={True,False}}
    buckets={hop:[q for q in pairs if q.startswith(str(hop)+'hop')] for hop in (2,3,4)}
    selected=[]
    for i,hop in enumerate((2,3,4)):
        selected.extend(sample_ids(buckets[hop], groups//3+(i<groups%3)))
    return [r for q in selected for r in sorted(pairs[q],key=lambda r:not r['answerable'])], {
        'archive':manifest['files']['musique.zip'],'available_rows':len(rows),'available_complete_pairs':len(pairs),
        'selected_group_ids':selected,'sample_seed':20260910,'scope':'per-question supplied context'}


def rank_metrics(qrels, ranking, *, ks=(10,100)):
    if len(ranking)!=len(set(ranking)): raise ValueError('Duplicate native IDs in ranking.')
    if not ks or any(type(k) is not int or k<1 for k in ks): raise ValueError('Invalid cutoff.')
    if any(type(v) is not int or v<0 for v in qrels.values()): raise ValueError('Invalid qrels.')
    positives={d for d,v in qrels.items() if v>0}; ideal=sorted(qrels.values(),reverse=True)
    result={}
    for k in ks:
        top=ranking[:k]; dcg=sum(qrels.get(d,0)/math.log2(i+2) for i,d in enumerate(top))
        idcg=sum(g/math.log2(i+2) for i,g in enumerate(ideal[:k]))
        result[f'ndcg@{k}']=dcg/idcg if idcg else None
        result[f'recall@{k}']=len(positives.intersection(top))/len(positives) if positives else None
        result[f'mrr@{k}']=next((1/(i+1) for i,d in enumerate(top) if d in positives),0.0) if positives else None
    return result


def aspect_metrics(aspects, ranking, *, k=10, alpha=.5):
    if not 0<=alpha<=1 or type(k) is not int or k<1 or len(ranking)!=len(set(ranking)):
        raise ValueError('Invalid aspect metric inputs.')
    if not aspects: return {'alpha_ndcg@'+str(k):None,'aspect_recall@'+str(k):None}
    total=sum(a['weight'] for a in aspects)
    weights={a['id']:a['weight']/total for a in aspects}
    mapping={d:a['id'] for a in aspects for d in a['supporting_docs']}
    counts=Counter(); dcg=0.0
    for i,d in enumerate(ranking[:k]):
        if d in mapping:
            a=mapping[d]; dcg+=weights[a]*(1-alpha)**counts[a]/math.log2(i+2); counts[a]+=1
    gains=sorted((weights[a['id']]*(1-alpha)**i for a in aspects for i in range(len(a['supporting_docs']))),reverse=True)
    ideal=sum(g/math.log2(i+2) for i,g in enumerate(gains[:k]))
    return {'alpha_ndcg@'+str(k):dcg/ideal if ideal else None,
            'aspect_recall@'+str(k):sum(weights[a] for a in counts)}


def score_ranking(dataset, qid, ranking, *, ks=(10,100)):
    if qid not in dataset.queries or not set(ranking)<={d['id'] for d in dataset.corpus}:
        raise ValueError('Unknown query/document in run.')
    return {**rank_metrics(dataset.qrels[qid],ranking,ks=ks),
            **(aspect_metrics(dataset.aspects[qid],ranking) if qid in dataset.aspects else {})}


def reference_metrics(qrels, rankings, *, ks=(10,100)):
    """Pinned trec_eval provider, explicit empty rankings and full query denominator."""
    import pytrec_eval
    if set(qrels)!=set(rankings): raise ValueError('Reference run must include every query.')
    evaluator=pytrec_eval.RelevanceEvaluator(qrels,{f'ndcg_cut.{k}' for k in ks}|{f'recall.{k}' for k in ks})
    scores={q:{d:float(len(ids)-i) for i,d in enumerate(ids)} for q,ids in rankings.items()}
    result=evaluator.evaluate(scores)
    return {q:{f'{m}@{k}':result.get(q,{}).get(f'{"ndcg_cut" if m=="ndcg" else "recall"}_{k}',0.0)
               for m in ('ndcg','recall') for k in ks} for q in qrels}
