from datetime import date
import typing as t

from bgpsyche.service.ext.ripe_as_names_countries import AsnTxtCountryCode

class ASFeaturesRaw(t.TypedDict):
    # - using this country code is probably fine, but
    # - we should somehow encode it as a category instead of coordinates,
    #   because then the model can more likely reflect political/administrative
    #   concerns such as a path being less likely to go through both north &
    #   south korea -> TODO: encode as category
    ripe_country: AsnTxtCountryCode
    # TODO: try replacing with our own customer cone computation from t1 mrt data
    as_rank: int
    rirstat_born: t.Optional[date]

    # - a smaller value could mean both that the AS is small or that is is a
    #   transit provider. content networks prob have larger prefixes because
    #   they operate more infra.
    # - TODO: try adding feature: how often did something change in rirstat?
    # - alternative: get from bgp routing tables
    #   -> TODO: try replacing with how many & how large prefixes from bgp
    rirstat_addr_count_v4: t.Optional[float]
    rirstat_addr_count_v6: t.Optional[float]



class PathFeatures(t.TypedDict):
    # Both length and is_valley_free may or may not be redundant. The model
    # could learn these properties on an AS level and giving them here
    # additionally could either be helpful or counter productive, we will have
    # to see.
    length: int
    is_valley_free: t.Optional[bool]