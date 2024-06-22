from datetime import datetime
import functools
from pprint import pformat
import json
from pathlib import Path
import typing as t
from abc import ABCMeta, abstractmethod, abstractproperty
import logging

from bgpsyche.caching.json_encoder import JSONDatetimeDecoder, JSONDatetimeEncoder
from bgpsyche.util.const import DATA_DIR
from bgpsyche.util.platform import get_func_module

_T = t.TypeVar('_T')
_CallableT = t.TypeVar('_CallableT', bound=t.Callable)
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
        custom: t.Any

    def _get_meta_defaults(self) -> _Meta:
        return {
            'last_updated': datetime.now(),
            'valid': False,
            'custom': None,
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
            custom: t.Any = None,
    ):
        args = locals()
        exists = self._get_meta() if self.__meta_path.exists() else {}
        defaults = self._get_meta_defaults()
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
                'custom'       : a_o_e_o_d('custom'),
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
            parents: t.List[Cacheable] = [],
            config_cache_path = _DEFAULT_CACHE_PATH,
            config_meta_path = Cache._DEFAULT_META_PATH,
    ) -> None:
        super().__init__(name, parents, config_meta_path)
        self.__getter           = getter
        self._config_cache_path = config_cache_path

    @abstractmethod
    def _serialize(self, data: _T) -> bytes: raise NotImplementedError()

    @abstractmethod
    def _deserialize(self, data: bytes) -> _T: raise NotImplementedError()

    @abstractproperty
    def _cache_path(self) -> str: raise NotImplementedError()

    def _on_miss(self) -> None:
        data = self.__getter()
        serdata = self._serialize(data)
        with open(self._cache_path, 'wb') as f: f.write(serdata)
        self._set_meta(valid=True, last_updated=datetime.now())

    def _retrieve(self) -> _T:
        with open(self._cache_path, 'rb') as f:
            return self._deserialize(f.read())


    @classmethod
    def decorate(cls, func: _CallableT) -> _CallableT:
        @functools.wraps(func)
        def serializable_file_cache_inner(*args, **kwargs):
            try:
                json.dumps(args); json.dumps(kwargs)
            except:
                _LOG.critical(
                    f'Could not serialize arguments as JSON: ' +
                    f'{pformat({"args": args, "kwargs": kwargs})}'
                )
                raise

            cache = cls(
                f'{get_func_module(func)}.{func.__name__}',
                lambda: func(*args, **kwargs)
            )

            params: t.Any = cache._get_meta()['custom'] or { 'args': [], 'kwargs': {} }
            if params['args'] != list(args):
                cache.invalidate()
                _LOG.info(f'Cache invalidated, {pformat((args, params["args"]))}')
            for key in kwargs.keys():
                if params['kwargs'][key] != kwargs[key]:
                    cache.invalidate()
                    _LOG.info(f'Cache invalidated, {pformat((args, params["args"]))}')

            if not cache._get_meta()['valid']:
                cache._set_meta(custom={ 'args': args, 'kwargs': kwargs })

            return cache.get()

        return t.cast(_CallableT, serializable_file_cache_inner)