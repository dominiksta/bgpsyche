from datetime import datetime
import json
from pathlib import Path
import typing as t
from abc import ABCMeta, abstractmethod
import logging

from caching.json_encoder import JSONDatetimeDecoder, JSONDatetimeEncoder
from util.const import DATA_DIR

_T = t.TypeVar('_T')
_LOG = logging.getLogger('caching')


class Cacheable(t.Generic[_T], metaclass=ABCMeta):
    # init
    # ----------------------------------------------------------------------

    def __init__(
            self,
            name: str,
            parents: t.List['Cacheable'] = [],
    ) -> None:
        self.name = name
        self.__parents = parents

    # user definition
    # ----------------------------------------------------------------------

    @abstractmethod
    def _on_miss(self) -> None: raise NotImplementedError()

    @abstractmethod
    def _retrieve(self) -> _T: raise NotImplementedError()

    @abstractmethod
    def check(self) -> bool: raise NotImplementedError()

    @abstractmethod
    def invalidate(self) -> None: raise NotImplementedError()

    # behaviour
    # ----------------------------------------------------------------------

    def ensure(self):
        for p in self.__parents: p.ensure()
        if not self.check():
            _LOG.info(f'Cache Miss: {self.name}')
            self._on_miss()
        else:
            _LOG.info(f'Cache Hit: {self.name}')


    def get(self):
        self.ensure()
        return self._retrieve()


class Cache(Cacheable[_T]):

    _DEFAULT_META_PATH = \
        DATA_DIR / 'cache' / f'{datetime.now().year}-{datetime.now().month}'

    # user definition
    # ----------------------------------------------------------------------

    def __init__(
            self,
            name: str,
            parents: t.List['Cacheable'] = [],
            config_meta_path = _DEFAULT_META_PATH,
    ) -> None:
        super().__init__(name, parents)

        self._config_meta_path = config_meta_path
        self._config_meta_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _retrieve(self) -> _T: raise NotImplementedError()

    # behaviour
    # ----------------------------------------------------------------------

    class _Meta(t.TypedDict):
        last_updated: datetime
        valid: bool

    def _get_meta_defaults(self) -> _Meta:
        return {
            'last_updated': datetime.now(),
            'valid': False,
        }

    @property
    def __meta_path(self) -> Path:
        return self._config_meta_path / f'{self.name}_meta.json'

    def _get_meta(self) -> _Meta:
        if not self.__meta_path.exists():
            self._set_meta(**self._get_meta_defaults())
        with open(self.__meta_path, 'r', encoding='UTF-8') as f:
            text = f.read()
            return json.loads(text, cls=JSONDatetimeDecoder)

    def _set_meta(
            self,
            valid: t.Optional[bool] = None,
            last_updated: t.Optional[datetime] = None,
    ):
        args = locals()
        exists = self._get_meta() if self.__meta_path.exists() else {}
        defaults = {
            'valid': False,
            'last_updated': datetime.now(),
        }
        # "args or exists or defaults"
        a_o_e_o_d = lambda k: (
            args[k] if args[k] is not None else (
                exists[k] if k in exists else (
                    defaults[k]
                )
            )
        )
        with open(self.__meta_path, 'w', encoding='UTF-8') as f:
            f.write(json.dumps({
                'valid'        : a_o_e_o_d('valid'),
                'last_updated' : a_o_e_o_d('last_updated'),
            }, cls=JSONDatetimeEncoder, indent=2))


    @property
    def last_updated(self) -> datetime:
        return self._get_meta()['last_updated']

    def check(self) -> bool:
        # _LOG.info(f'CHECK: {}')
        return (
            self.__meta_path.exists()
            and self._get_meta()['valid']
        )

    def invalidate(self) -> None:
        _LOG.info(f'Cache Invalidate: {self.name}')
        self._set_meta(valid=False)



class SerializableFileCache(Cache[_T]):

    _DEFAULT_CACHE_PATH = Cache._DEFAULT_META_PATH

    def __init__(
            self,
            name: str,
            getter: t.Callable[[], _T],
            serialize: t.Callable[[_T], bytes],
            deserialize: t.Callable[[bytes], _T],
            parents: t.List[Cacheable] = [],
            config_cache_path = _DEFAULT_CACHE_PATH,
            config_meta_path = Cache._DEFAULT_META_PATH,
    ) -> None:
        super().__init__(name, parents, config_meta_path)
        self.__getter           = getter
        self._config_cache_path = config_cache_path
        self._serialize         = serialize
        self._deserialize       = deserialize


    @property
    def _cache_path(self):
        return self._config_cache_path / f'{self.name}_data.json'

    def _on_miss(self) -> None:
        data = self.__getter()
        serdata = self._serialize(data)
        with open(self._cache_path, 'wb') as f: f.write(serdata)
        self._set_meta(valid=True, last_updated=datetime.now())

    def _retrieve(self) -> _T:
        with open(self._cache_path, 'rb') as f:
            return self._deserialize(f.read())