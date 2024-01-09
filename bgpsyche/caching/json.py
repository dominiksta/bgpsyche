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
            parents=parents,
            config_meta_path=config_meta_path, 
            config_cache_path=config_cache_path, 
        )

    @property
    def _cache_path(self):
        return self._config_cache_path / f'{self.name}_data.json'

    def _serialize(self, data):
        return json.dumps(data).encode('UTF-8')

    def _deserialize(self, data):
        return json.loads(data.decode('UTF-8'))
