"""Strict MuSiQue output mapping and independently calculated paired metrics."""
from collections import Counter, defaultdict
import json
import re
import string


OUTPUT_INSTRUCTION = ('\n\nUse only evidence in this knowledge base. Return a JSON object with exactly '
    '"answer" (a short answer string), "answerable" (boolean), and "support_sources" '
    '(the exact .md filenames supporting the answer). If the supplied knowledge base '
    'does not contain sufficient evidence, set answerable to false and explain no facts beyond that evidence.')


def parse_prediction(text, sources, *, stopped='final'):
    failure={'predicted_answer':'','predicted_answerable':None,'predicted_support_idxs':[]}
    if stopped!='final': return failure,'execution_did_not_finish'
    try:
        raw=text.strip()
        if raw.startswith('```') and raw.endswith('```'):
            raw=raw.split('\n',1)[1].rsplit('```',1)[0].strip()
        def unique(pairs):
            obj={}
            for k,v in pairs:
                if k in obj: raise ValueError('Duplicate output field.')
                obj[k]=v
            return obj
        obj=json.loads(raw,object_pairs_hook=unique)
        if (set(obj)!={'answer','answerable','support_sources'} or not isinstance(obj['answer'],str)
                or type(obj['answerable']) is not bool or not isinstance(obj['support_sources'],list)
                or any(not isinstance(s,str) or s not in sources for s in obj['support_sources'])
                or len(obj['support_sources'])!=len(set(obj['support_sources']))):
            raise ValueError('Invalid answer/support schema.')
        return {'predicted_answer':obj['answer'],'predicted_answerable':obj['answerable'],
                'predicted_support_idxs':[sources[s] for s in obj['support_sources']]},None
    except (ValueError,TypeError,AttributeError,KeyError) as e:
        return failure,type(e).__name__+': '+str(e)


def _normalize(text):
    no_punctuation=''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',no_punctuation).split())


def answer_f1(prediction, aliases):
    p=Counter(_normalize(prediction).split()); values=[]
    for a in aliases:
        g=Counter(_normalize(a).split()); denominator=sum(p.values())+sum(g.values())
        values.append(2*sum((p & g).values())/denominator if denominator else 1.0)
    return max(values)


def support_f1(predicted, gold):
    p,g=set(predicted),set(gold)
    return 2*len(p & g)/(len(p)+len(g)) if p or g else 1.0


def musique_metrics(gold, predictions):
    if len(gold)!=len(predictions): raise ValueError('Missing predictions.')
    groups=defaultdict(list); answers=[]; supports=[]; exact=[]; correctness=[]
    for g,p in zip(gold,predictions):
        if g['id']!=p['id']: raise ValueError('Prediction order differs.')
        correct=p['predicted_answerable'] is g['answerable']; correctness.append(correct)
        af=sf=em=None
        if g['answerable']:
            aliases=[g['answer'],*g['answer_aliases']]
            af=answer_f1(p['predicted_answer'],aliases)
            sf=support_f1(p['predicted_support_idxs'],[r['idx'] for r in g['paragraphs'] if r['is_supporting']])
            em=float(any(_normalize(p['predicted_answer'])==_normalize(a) for a in aliases))
            answers.append(af);supports.append(sf);exact.append(em)
        groups[g['id']].append((g['answerable'],correct,af,sf))
    joint_answer=[];joint_support=[]
    for rs in groups.values():
        if len(rs)!=2 or {r[0] for r in rs}!={True,False}: raise ValueError('Incomplete MuSiQue pair.')
        good=all(r[1] for r in rs); answered=next(r for r in rs if r[0])
        joint_answer.append(answered[2] if good else 0.0);joint_support.append(answered[3] if good else 0.0)
    mean=lambda xs:sum(xs)/len(xs) if xs else None
    return {'answer_f1':mean(answers),'answer_em':mean(exact),'support_f1':mean(supports),
            'group_answer_sufficiency_f1':mean(joint_answer),'group_support_sufficiency_f1':mean(joint_support),
            'answerability_accuracy':mean(correctness),'groups':len(groups),'rows':len(gold)}
