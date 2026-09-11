"""Bind failed Agent tool arguments to actual registered documents and prior hits."""
import argparse
import json
from pathlib import Path
from arkb.evaluation.external import digest,load_external,read_jsonl,verify_checksums,write_json,ExternalDataset
from arkb.knowledge.models import _document_id
from arkb.knowledge.chunking import _sections


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    parser.add_argument('--dataset',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify_checksums(args.run)
    if args.output.exists():raise ValueError('Use a new audit path.')
    if json.loads((args.run/'experiment.json').read_text())['status']!='completed':raise ValueError('Incomplete run.')
    protocol=json.loads((args.run/'protocol.json').read_text());mu=protocol['track']=='musique'
    if mu:
        selected=args.run/'selected.json'
        if digest(selected)!=protocol['selection_sha256']:raise ValueError('Selection changed.')
        cases=json.loads(selected.read_text())['rows']
    else:
        data=load_external(args.dataset)
        if digest(args.dataset/'manifest.json')!=protocol['dataset_manifest_sha256']:raise ValueError('Dataset changed.')
        notes={_document_id(data.name,n.source):n for n in data.notes()}
    failures=[]
    for row in read_jsonl(args.run/'rows.jsonl'):
        if not row['error']:continue
        if mu:
            case=cases[row['case_index']]
            data=ExternalDataset(f'musique-{row["case_index"]}',[{'id':str(p['idx']),'title':p['title'],'text':p['paragraph_text']} for p in case['paragraphs']],{'q':case['question']},{'q':{}},{},{})
            notes={_document_id('p4-musique',n.source):n for n in data.notes()}
        report=(row['result'] or {}).get('observation',{});previous=[]
        for tool in report.get('tools',[]):
            if tool.get('error'):
                args_read=tool['arguments'];item={'case_index':row['case_index'],'id':row['id'],
                    'tool':tool['name'],'arguments':args_read,'error':tool['error']}
                if tool['name']=='read':
                    document_id=args_read.get('document_id')
                    target=notes.get(document_id) if isinstance(document_id,str) else next((n for n in notes.values() if n.source==args_read.get('source')),None)
                    item['target_document_exists']=target is not None
                    if target:
                        item.update(source=target.source,document_revision=target.document_revision,body_characters=len(target.content))
                        for key in ('start_char','end_char'):
                            if args_read.get(key) is not None:
                                item[key+'_invalid_type']=type(args_read[key]) is not int
                                if type(args_read[key]) is int:item[key+'_exceeds_body']=args_read[key]>len(target.content)
                        if isinstance(args_read.get('section_id'),str):
                            section=args_read['section_id'];item['section_exists_in_target']=section in {s.section_id for s in _sections(target)}
                            owners=sorted({h['document_id'] for h in previous if h.get('section_id')==section})
                            item['prior_returned_document_ids_for_section']=owners
                            item['mixed_document_and_section_from_different_hits']=bool(owners) and args_read.get('document_id') not in owners
                failures.append(item)
            raw=tool.get('raw_result')
            if raw is not None and tool['delivered_to_conversation']:
                previous.extend([raw['result']] if tool['name']=='read' else raw['results'])
    write_json(args.output,{'run':args.run.name,'rows_sha256':digest(args.run/'rows.jsonl'),
        'audit_sha256':digest(Path(__file__)),'tool_failures':failures,
        'scope':'Descriptive checks of retained failed tool arguments against the registered source bodies and previously delivered hit metadata. This is not an answer-quality judgment.',
        'release_eligible':False})


if __name__=='__main__':main()
