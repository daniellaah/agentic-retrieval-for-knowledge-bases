"""Explicit software/runtime provenance for local model experiments."""
from importlib.metadata import version,PackageNotFoundError
import json
import os
import platform
import sys
from urllib.request import urlopen


def runtime_environment(runtime):
    packages={}
    for name in ('numpy','ollama','qdrant-client','tokenizers','pyarrow','pytrec-eval-terrier','torch','transformers'):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    with urlopen(runtime.config.host.rstrip('/')+'/api/version',timeout=10) as response:
        ollama=json.load(response)
    with urlopen(runtime.config.qdrant_url.rstrip('/'),timeout=10) as response:
        qdrant=json.load(response)
    return {'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'cpu_count':os.cpu_count(),
        'packages':packages,'ollama_server_version':ollama['version'],
        'qdrant_server':{k:qdrant.get(k) for k in ('version','commit')},
        'thread_settings':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','TOKENIZERS_PARALLELISM')}}
