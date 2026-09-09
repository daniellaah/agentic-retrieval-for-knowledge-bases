"""Compatibility imports; shared chunking lives in :mod:`arkb.chunking`."""

from arkb.chunking import chunk_notes, whole_note_chunks
from arkb.schema import Chunk, Note

__all__ = ["Chunk", "Note", "chunk_notes", "whole_note_chunks"]
