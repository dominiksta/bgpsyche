import pickle
import typing as t

import bgpsyche.caching.base as base

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
        self.__pickle_protocol = pickle_protocol
        super().__init__(
            name=name,
            getter=getter,
            parents=parents,
            config_meta_path=config_meta_path, 
            config_cache_path=config_cache_path, 
        )

    @property
    def _cache_path(self):
        return self._config_cache_path / f'{self.name}_data.pickle'

    def _serialize(self, data):
        return pickle.dumps(data, protocol=self.__pickle_protocol)

    def _deserialize(self, data):
        return pickle.loads(data)