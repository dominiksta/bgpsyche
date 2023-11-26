import typing as t

ISODateTimeStr = str

class Network(t.TypedDict):
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
    info_type: t.Literal[
        'Not Disclosed', 'NSP', 'Content', 'Cable/DSL/ISP', 'Enterprise',
        'Educational/Research', 'Non-Profit', 'Route Server', 'Network Services',
        'Route Collector', 'Government',
    ]
    info_prefixes4: int
    info_prefixes6: int
    info_traffic: str
    info_ratio: str
    info_scope: str
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
    policy_general: t.Literal[
        'Open', 'Selective', 'Restrictive', 'No'
    ]
    policy_locations: t.Literal[
        'Not Required', 'Preferred',
        'Required - US', 'Required - EU', 'Required - International', 
    ]
    policy_ratio: bool
    policy_contracts: t.Literal['Not Required', 'Private Only', 'Required']
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