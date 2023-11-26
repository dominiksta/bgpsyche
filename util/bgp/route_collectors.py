ROUTE_COLLECTORS = {
    'ripe_ris'              : 12654,
    'routeviews'            : 6447,
    # Note that PCH provides other services under AS42, only AS3846 is their
    # research network.
    'packet_clearing_house' : 3856,
}

def is_route_collector(asn: int) -> bool:
    return asn in ROUTE_COLLECTORS.values()
