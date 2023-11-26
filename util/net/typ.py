import typing as t
from ipaddress import IPv4Address, IPv4Network, IPv6Address, IPv6Network

IPAddress = t.Union[IPv4Address, IPv6Address]
IPNetwork = t.Union[IPv4Network, IPv6Network]

class HopResponse(t.TypedDict):
    addr: t.Union[IPAddress, t.Literal['*']]
    rtt_ms: float

class Traceroute(t.TypedDict):
    src: IPAddress
    dst: IPAddress
    hops: t.List[HopResponse]
