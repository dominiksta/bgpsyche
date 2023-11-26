import json
import typing as t

import bgpsyche.caching.base as base

JSONSerializable = t.TypeVar(
    'JSONSerializable',
    bound=t.Union[t.List, t.Dict, str, int, float]
)

class JSONFileCache(base.SerializableFileCache[JSONSerializable]):
    def __init__(
            self,
            name: str,
            getter: t.Callable[[], JSONSerializable],
            parents: t.List[base.Cacheable] = [],
            config_cache_path = base.SerializableFileCache._DEFAULT_CACHE_PATH,
            config_meta_path = base.Cache._DEFAULT_META_PATH,
    ) -> None:
        super().__init__(
            name=name,
            getter=getter,
            serialize=lambda v: json.dumps(v).encode('UTF-8'),
            deserialize=lambda b: json.loads(b.decode('UTF-8')),
            parents=parents,
            config_meta_path=config_meta_path, 
            config_cache_path=config_cache_path, 
        )
