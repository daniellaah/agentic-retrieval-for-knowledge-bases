"""Offline official aspect metric audit."""
import ast
from collections import defaultdict
import json
from pathlib import Path

from arkb.evaluation.external import load_external, read_jsonl, aspect_metrics, digest, write_json


def reference_functions(inputs):
    folder=inputs/'official/bright-pro/retrieval/evaluation'
    lock=json.loads((Path(__file__).resolve().parents[1]/'experiments/p4-downloads.lock.json').read_text())
    names={'_get_aspect_weight','compute_alpha_dcg_at_k','compute_alpha_idcg_at_k','compute_weighted_aspect_recall_at_k'}
    nodes={}
    for path in sorted(folder.glob('*.py')):
        expected=lock['files'][path.relative_to(inputs).as_posix()]['sha256']
        if digest(path)!=expected:raise ValueError('Unreviewed official metric source bytes.')
        tree=ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in names:nodes[node.name]=node
    if set(nodes)!=names:raise ValueError('Missing official metric functions.')
    # Only the reviewed pure metric functions, with postponed type annotations;
    # dataset loaders, imports, CLI and optional export code are not executed.
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes.values()],type_ignores=[])
    ast.fix_missing_locations(module); namespace={}
    exec(compile(module,'reviewed-bright-pro-metric-functions','exec'),namespace)
    return namespace


def aspect_audit(root):
    inputs=root/'p4-inputs'; functions=reference_functions(inputs); checks=0;maximum=0.0;collisions={}
    for name in ('bright-stackoverflow','bright-robotics'):
        data=load_external(root/'p4-data'/name);mapping={};weights={};seen=defaultdict(set)
        for aspects in data.aspects.values():
            total=sum(a['weight'] for a in aspects)
            for a in aspects:
                weights[a['id']]=a['weight']/total
                for d in a['supporting_docs']:seen[d].add(a['id']);mapping[d]=a['id']
        collisions[name]={d:sorted(a) for d,a in seen.items() if len(a)>1}
        if collisions[name]:raise ValueError('Official global aspect map is ambiguous: '+name)
        fixtures=[]
        for qid in data.queries:
            fixtures.extend([(qid,[]),(qid,sorted(data.qrels[qid])),(qid,list(reversed(sorted(data.qrels[qid]))))])
        for file in root.glob(f'p4-{name}-*/rows.jsonl'):
            for row in read_jsonl(file):
                if 'ranking' in row:fixtures.append((row['qid'],row['ranking']))
        for qid,ranking in fixtures:
            gold=list(data.qrels[qid]);ours=aspect_metrics(data.aspects[qid],ranking)
            dcg=functions['compute_alpha_dcg_at_k'](ranking,gold,mapping,weights,.5,10)
            ideal=functions['compute_alpha_idcg_at_k'](gold,mapping,weights,.5,10)
            ar=functions['compute_weighted_aspect_recall_at_k'](ranking,gold,mapping,weights,10)
            maximum=max(maximum,abs(ours['alpha_ndcg@10']-dcg/ideal),abs(ours['aspect_recall@10']-ar));checks+=2
    if maximum>1e-9:raise ValueError('Official aspect scorer mismatch.')
    return {'comparisons':checks,'maximum_absolute_error':maximum,'cross_query_aspect_collisions':collisions,
            'official_files':{p.relative_to(inputs).as_posix():digest(p) for p in (inputs/'official/bright-pro').rglob('*.py')}}


if __name__=='__main__':
    root=Path('evaluation/results')
    write_json(root/'p4-regression/aspect-reference.json',aspect_audit(root))
    print('Official aspect metrics audit completed.')
