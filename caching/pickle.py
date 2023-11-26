import pickle
import typing as t

import caching.base as base

_T = t.TypeVar('_T')

class PickleFileCache(base.SerializableFileCache[_T]):
    def __init__(
            self,
            name: str,
            getter: t.Callable[[], _T],
            parents: t.List[base.Cacheable] = [],
            config_cache_path = base.SerializableFileCache._DEFAULT_CACHE_PATH,
            config_meta_path = base.Cache._DEFAULT_META_PATH,
            pickle_protocol = pickle.DEFAULT_PROTOCOL,
    ) -> None:
        super().__init__(
            name=name,
            getter=getter,
            serialize=lambda v: pickle.dumps(v, protocol=pickle_protocol),
            deserialize=lambda b: pickle.loads(b),
            parents=parents,
            config_meta_path=config_meta_path, 
            config_cache_path=config_cache_path, 
        )

    @property
    def _cache_path(self):
        return self._config_cache_path / f'{self.name}_data.pickle'
