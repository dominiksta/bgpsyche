import subprocess
import distutils.spawn
from bgpsyche.caching.json import JSONFileCache

def whois_raw(query: str) -> str:
    if not distutils.spawn.find_executable("whois"):
        raise NotImplementedError()
    
    return JSONFileCache(
        f'whois_{query}',
        lambda: subprocess.check_output(['whois', query]).decode('UTF-8'),
    ).get()