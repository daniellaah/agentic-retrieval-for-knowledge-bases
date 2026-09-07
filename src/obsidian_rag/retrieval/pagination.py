"""Bounded pages for deterministic searches, with query- and snapshot-bound cursors."""

import base64
from dataclasses import replace
from itertools import islice
import json

from obsidian_rag.knowledge_base.identity import fingerprint_config, require_integer, require_text
from .models import SearchResponse


def paginate(results, *, query, method, scope, limit, cursor, parameters):
    require_text(query, 'query')
    require_integer(limit, 'limit', minimum=1)
    identity = fingerprint_config(dict(query=query, method=method, vault_id=scope.vault_id,
                                      snapshot_id=scope.snapshot_id, paths=list(scope.paths), parameters=parameters))
    offset = 0
    if cursor is not None:
        try:
            if not isinstance(cursor, str) or len(cursor) > 1024:
                raise ValueError()
            data = json.loads(base64.b64decode(cursor, altchars=b'-_', validate=True))
            if (not isinstance(data, dict) or set(data) != {'request', 'offset'}
                    or data['request'] != identity or type(data['offset']) is not int or data['offset'] <= 0):
                raise ValueError()
            offset = data['offset']
        except (ValueError, TypeError, UnicodeError) as error:
            raise ValueError('Invalid cursor or cursor from another query, scope, or snapshot.') from error
    page = list(islice(results, offset, offset + limit + 1))
    has_more = len(page) > limit
    continuation = None
    if has_more:
        continuation = base64.urlsafe_b64encode(json.dumps({'request': identity, 'offset': offset + limit}).encode()).decode()
    items = tuple(replace(hit, rank=rank) for rank, hit in enumerate(page[:limit], 1))
    return SearchResponse(query, method, scope, items, limit, has_more=has_more, next_cursor=continuation)
