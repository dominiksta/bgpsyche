import typing as t

from bgpsyche.service.ext import peeringdb
from bgpsyche.service.ext.asdb import NAICSLiteSelection, get_asdb_primary

ASCategory = t.Literal[
    # almost the same as peeringdb.NetworkType, except 'Not Disclosed' becomes
    # 'Unknown' and 'NSP' & 'Cable/DSL/ISP' are merged into 'Transit/Access'
    # much like in https://www.caida.org/catalog/datasets/as-classification/
    # 'Not Disclosed',
    'Unknown',
    # 'NSP',
    # 'Cable/DSL/ISP',
    'Transit/Access',
    'Content',
    'Enterprise',
    'Educational/Research',
    'Non-Profit',
    'Route Server',
    'Network Services',
    'Route Collector',
    'Government',
]

AS_CATEGORY: t.Set[ASCategory] = set(t.get_args(ASCategory))

_FROM_PEERINGDB: t.Dict[peeringdb.NetworkType, ASCategory] = {
    'Not Disclosed': 'Unknown',
    'Cable/DSL/ISP': 'Transit/Access',
    'NSP': 'Transit/Access',
    'Content': 'Content',
    'Enterprise': 'Enterprise',
    'Educational/Research': 'Educational/Research',
    'Non-Profit': 'Non-Profit',
    'Route Collector': 'Route Collector',
    'Network Services': 'Network Services',
    'Route Server': 'Route Server',
    'Government': 'Government',
}

_FROM_ASDB: t.Dict[
    NAICSLiteSelection, ASCategory
] = {
    # Layer 1
    # ----------------------------------------------------------------------
    'Media, Publishing, and Broadcasting': 'Content',
    'Finance and Insurance': 'Enterprise',
    # While there is an Educational/Research type in peeringdb, it is likely
    # that ASes listed in peeringdb will do specifically internet research while
    # ASes listed as research here may do other types of research. Government
    # does of course not map exactly, but for prediciting routes it should
    # hopefully be close enough.
    'Education and Research': 'Government',
    'Service': 'Enterprise',
    'Agriculture, Mining, and Refineries (Farming, Greenhouses, Mining, Forestry, and Animal Farming)': 'Enterprise',
    'Community Groups and Nonprofits': 'Non-Profit',
    'Construction and Real Estate': 'Enterprise',
    'Museums, Libraries, and Entertainment': 'Enterprise',
    'Utilities (Excluding Internet Service)': 'Government',
    'Health Care Services': 'Government',
    'Travel and Accommodation': 'Enterprise',
    'Freight, Shipment, and Postal Services': 'Enterprise',
    'Government and Public Administration': 'Government',
    'Retail Stores, Wholesale, and E-commerce Sites': 'Enterprise',
    'Manufacturing': 'Enterprise',
    'Other': 'Unknown',
    'Unknown': 'Unknown',

    # Computer and Technology
    # ----------------------------------------------------------------------

    'Computer and IT - Internet Service Provider (ISP)': 'Transit/Access',
    'Computer and IT - Phone Provider': 'Transit/Access',
    'Computer and IT - Hosting, Cloud Provider, Data Center, Server Colocation': 'Content',
    'Computer and IT - Computer and Network Security': 'Enterprise',
    'Computer and IT - Software Development': 'Enterprise',
    'Computer and IT - Technology Consulting Services': 'Enterprise',
    'Computer and IT - Satellite Communication': 'Transit/Access',
    'Computer and IT - Search': 'Enterprise',
    'Computer and IT - Internet Exchange Point (IXP)': 'Route Server',
    'Computer and IT - Other': 'Unknown',
    'Computer and IT - Unknown': 'Unknown',
}

def get_as_category(asn: int) -> ASCategory:
    pdb_entry = peeringdb.Client.get_network_by_asn(asn)
    # if pdb_entry: print(asn, pdb_entry.info_type)
    assert pdb_entry is None or pdb_entry.info_type != '', \
        { 'asn': asn, 'pdb_entry': pdb_entry.__dict__ }
    cat = _FROM_PEERINGDB[pdb_entry.info_type] \
        if pdb_entry is not None else _FROM_ASDB[get_asdb_primary(asn)]
    assert cat in AS_CATEGORY
    return cat



if __name__ == '__main__':
    show = {
        3320: 'DTAG',
        39063: 'Leitwert',
        51402: 'COM-IN',
        51378: 'Klinikum Ingolstadt',
        16509: 'Amazon',
        64199: 'TCPShield (DDOS Protection)',
        13335: 'Cloudflare',
        17374: 'Walmart',
        32934: 'Meta (Zuckbook)',
        8075: 'Micro$oft',
        6695: 'DE-CIX Frankfurt Route Servers',
    }
    for asn, name in show.items():
        print(f'{asn: >6} ({name: >40}): {get_as_category(asn)}')