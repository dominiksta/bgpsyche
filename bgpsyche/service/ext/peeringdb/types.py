from dataclasses import dataclass
import typing as t

ISODateTimeStr = str

NetworkType = t.Literal[
    'Not Disclosed', 'NSP', 'Content', 'Cable/DSL/ISP', 'Enterprise',
    'Educational/Research', 'Non-Profit', 'Route Server', 'Network Services',
    'Route Collector', 'Government',
]

NetworkPolicyGeneral = t.Literal['Open', 'Selective', 'Restrictive', 'No']

NetworkPolicyLocations = t.Literal[
    'Not Required', 'Preferred',
    'Required - US', 'Required - EU', 'Required - International',
]

NetworkPolicyContracts = t.Literal['Not Required', 'Private Only', 'Required']

NetworkTraffic = t.Literal[
    '', '0-20Mbps', '20-100Mbps', '100-1000Mbps', '1-5Gbps', '5-10Gbps',
    '10-20Gbps', '20-50Gbps', '50-100Gbps', '100-200Gbps', '200-300Gbps',
    '300-500Gbps', '500-1000Gbps', '1-5Tbps', '5-10Tbps', '10-20Tbps',
    '20-50Tbps' '50-100Tbps' '100+Tbps'
]

NetworkRatio = t.Literal[
    '', 'Not Disclosed', 'Heavy Outbound', 'Mostly Outbound', 'Balanced',
    'Mostly Inbound', 'Heavy Inbound'
]

NetworkScope = t.Literal[
    '', 'Not Disclosed', 'Regional', 'North America', 'Asia Pacific', 'Europe',
    'South America', 'Africa', 'Australia', 'Middle East', 'Global'
]

@dataclass
class Network:
    id: int
    org_id: int
    name: str
    aka: str
    name_long: str
    website: str
    social_media: t.Dict[str, str]
    asn: int
    looking_glass: str
    route_server: str
    irr_as_set: str
    info_type: NetworkType
    info_prefixes4: int
    info_prefixes6: int
    info_traffic: NetworkTraffic
    info_ratio: NetworkRatio
    info_scope: NetworkScope
    info_unicast: bool
    info_multicast: bool
    info_ipv6: bool
    info_never_via_route_servers: bool
    ix_count: int
    fac_count: int
    notes: str
    netixlan_updated: ISODateTimeStr
    netfac_updated: ISODateTimeStr
    poc_updated: ISODateTimeStr
    policy_url: str
    policy_general: NetworkPolicyGeneral
    policy_locations: NetworkPolicyLocations
    policy_ratio: bool
    policy_contracts: NetworkPolicyContracts
    netfac_set: t.List[t.Dict] # HACK
    netixlan_set: t.List[t.Dict]
    poc_set: t.List[t.Dict]
    allow_ixp_update: bool
    status_dashboard: str
    rir_status: str
    rir_status_updated: ISODateTimeStr
    created: ISODateTimeStr
    updated: ISODateTimeStr
    status: str