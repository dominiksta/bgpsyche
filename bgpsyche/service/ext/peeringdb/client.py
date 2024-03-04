import typing as t

import peeringdb.resource as _resource
resource = t.cast(t.Any, _resource) # shut up pyright
import peeringdb.client
from bgpsyche.logging_config import logging_setup
from bgpsyche.util.const import ENV
from .types import Network
from bgpsyche.util.const import DATA_DIR, ENV

_PDB_DIR = DATA_DIR / 'peeringdb'

# adapted from default config when running `peeringdb config set -n`
_PDB_CONFIG = {
    'orm': {
        'backend': 'django_peeringdb',
        'database': {
            'engine': 'sqlite3',
            'host': '',
            'name': _PDB_DIR / 'peeringdb.sqlite3',
            'password': '',
            'port': 0,
            'user': '',
        },
        'migrate': True,
        'secret_key': ''
    },
    'sync': {
        'api_key': ENV['peeringdb']['api_key'],
        'cache_dir': _PDB_DIR / 'cache',
        'cache_url': 'https://public.peeringdb.com',
        'only': [],
        'password': '',
        'strip_tz': 1,
        'timeout': 0,
        'url': 'https://www.peeringdb.com/api',
        'user': '',
    }
}

_pdb = peeringdb.client.Client(cfg=_PDB_CONFIG)
# peeringdb overwrites logging config.
# see https://github.com/peeringdb/peeringdb-py/issues/67
logging_setup()

# we need to import this after initializing the client
import django_peeringdb.models.concrete as models

class Client():

    @staticmethod
    def sync():
        _pdb.updater.update_all([
            resource.Organization,              # org
            resource.Campus,                    # campus
            resource.Facility,                  # fac
            resource.Network,                   # net
            resource.InternetExchange,          # ix
            resource.Carrier,                   # carrier
            resource.CarrierFacility,           # carrierfac
            resource.InternetExchangeFacility,  # ixfac
            resource.InternetExchangeLan,       # ixlan
            resource.InternetExchangeLanPrefix, # ixpfx
            resource.NetworkFacility,           # netfac
            resource.NetworkIXLan,              # netixlan
            resource.NetworkContact,            # poc
        ])

    @staticmethod
    def get_network_by_asn(asn: int) -> t.Optional[Network]:
        resp = _pdb.all(resource.Network).filter(asn=asn)
        ret = resp[0] if len(resp) != 0 else None

        if ret is not None:
            # no idea
            if ret.info_type == '': ret.info_type = 'Not Disclosed'

        return ret


    class _RouteServer(t.TypedDict):
        asn: int
        name: str
        peeringdb_org_id: int

    @staticmethod
    def get_route_servers() -> t.List[_RouteServer]:
        data = t.cast(t.List[Network], _pdb.all(Network).filter(info_type='Route Server'))
        return list(
            {
                'asn': net.asn,
                'name': net.name,
                'peeringdb_org_id': net.org_id,
            } for net in data
        )
