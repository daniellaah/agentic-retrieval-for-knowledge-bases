from arkb.knowledge.models import Note, ChunkRecord
from arkb.knowledge.chunking import whole_note_chunks


def citation_fixture():
    from arkb.generation.models import ContextConfig, GenerationCounter
    from arkb.generation.context import build_context
    from arkb.retrieval.semantic import snapshot_result
    note = Note('Title', 'Only small datasets were faster.', 'a.md')
    chunk = whole_note_chunks([note])[0]
    counter = GenerationCounter('test', 'chars', lambda m: 10 + sum(len(x['content']) for x in m))
    return build_context('Which datasets were faster?', [snapshot_result(ChunkRecord.from_note(chunk, note=note, vault_id='v'), .9, 'snapshot')],
                          config=ContextConfig(), counter=counter, citation_mode='structured')
