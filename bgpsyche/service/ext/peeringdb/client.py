import typing as t
import urllib3
from urllib3.connection import HTTPException
from urllib3.response import BaseHTTPResponse
import logging

from bgpsyche.caching.json import JSONFileCache
from bgpsyche.util.const import ENV
from .types import Network

_API_BASE_URL = 'https://www.peeringdb.com/api'

_T = t.TypeVar('_T')
_LOG = logging.getLogger(__name__)

def _req_peeringdb(
        method: str, path: str,
        fields: t.Dict[str, t.Any] = {},
        headers: t.Dict[str, t.Any] = {},
) -> BaseHTTPResponse:
    # note that we could probably also just download a dump of all peeringdb
    # from caida: https://publicdata.caida.org/datasets/peeringdb/

    _LOG.info(f'PeeringDB API: {method} {path}')
    
    resp = urllib3.request(
        method, f'{_API_BASE_URL}{path}', fields=fields,
        retries=3, timeout=60,
        headers={
            **headers,
            'Authorization': 'Api-Key ' + ENV['peeringdb']['api_key'],
            'Content-Type': 'application/json',
        }
    )
    if str(resp.status).startswith('5'): raise HTTPException({
            'status': resp.status,
            'msg': resp.data,
    })
    return resp

class _CachedHTTPResponse(t.TypedDict):
    status: int
    json: t.Any


def _cache(id: str, getter: t.Callable[[], _T]) -> _T:
    # HACK: we should implement some cache eviction strategy
    return JSONFileCache(f'peeringdb_{id}', t.cast(t.Any, getter)).get()

def _cache_request(
        id: str, path: str, fields: t.Dict[str, t.Any] = {},
) -> _CachedHTTPResponse:
    def _request() -> _CachedHTTPResponse:
        resp = _req_peeringdb('GET', path, fields=fields)
        return { 'status': resp.status, 'json': resp.json() }

    return _cache(id, _request)


class Client():

    @staticmethod
    def get_network_by_asn(asn: int) -> t.Optional[Network]:
        resp = _cache_request(f'network_{asn}', '/net', {'asn': asn})
        if resp['status'] == 404: return None
        else: return resp['json']['data'][0]


    class _RouteServer(t.TypedDict):
        asn: int
        name: str
        peeringdb_org_id: int

    @staticmethod
    def get_route_servers(limit = 100_000) -> t.List[_RouteServer]:
        resp = _cache_request(
            'route_servers', '/net', fields={
                'info_type': 'Route Server',
                'limit': limit,
            }
        )
        data: t.List[Network] = resp['json']['data']
        return list(
            {
                'asn': net['asn'],
                'name': net['name'],
                'peeringdb_org_id': net['org_id'],
            } for net in data
        )
