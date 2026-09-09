from arkb.knowledge.models import Chunk, Note, ChunkRecord
from arkb.retrieval.semantic import snapshot_result


def source_hit(start, end, *, text='abcdefghijklmnop', source='a.md', index=0,
               version='v1', vault='vault', score=.8):
    from arkb.knowledge.models import Note
    from arkb.knowledge.models import ChunkRecord
    note = Note('Title', text, source)
    chunk = Chunk(text[start:end], note.title, source, index, start, end)
    record = ChunkRecord.from_note(chunk, note=note, vault_id=vault)
    return snapshot_result(record, score, version)
