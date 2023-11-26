import typing as t
from collections import defaultdict
from functools import reduce
from itertools import permutations

from util.bgp.relationship import Source2Sink2Rel

# manually determined
TIER1 = {
    174,  # Cogent
    1239, # Sprint
    1299, # Telia
    2914, # NTT
    3257, # GTT
    3320, # DTAG
    3356, # Level3
    3491, # PCCW
    6453, # Tata
    6461, # Zayo
    6762, # Seabone
    6830, # Liberty
    6939, # Hurricane
    7018, # AT&T
}

# manually determined
SIBLINGS: t.Dict[int, t.Set[int]] = {
    3320  : {5588, 5606},                                     # DTAG
    1299  : {3301},                                           # Telia
    3356  : {3549, 9057, 11415, 58682},                       # Level3
    3257  : {5580, 8928},                                     # GTT
    2914  : {4713},                                           # NTT
    4755  : {6421, 6453, 45820},                              # Tata
    2386  : {2686, 2687, 4466, 6389, 7018, 17224, 17225},     # AT&T
    1221  : {4637},                                           # Telstra
    3352  : {7418, 10429, 12956, 18881, 22927, 26599, 27699}, # Telefonica
    1273  : {3209, 6739, 12430, 15924, 33915, 55410},         # Vodafone
    33891 : {201011},                                         # Core-Backbone
    7015  : {7922, 33657, 33491},                             # Comcast
    701   : {6167},                                           # Verizon
    16625 : {20940, 32787, 35994},                            # Akamai
    9498  : {45609},                                          # Bharti
    3758  : {7473, 7474},                                     # SingTel
    7303  : {10318, 10481},                                   # Telecom Argentinia
    27299 : {55195},                                          # CIRA
    7843  : {10796, 20115},                                   # Charter
    3722  : {36236},                                          # NetActuate
    292   : {293},                                            # ESNET
    57    : {217},                                            # University of Minnesota
    680   : {1275},                                           # DFN
    5511  : {8376},                                           # Orange
    12880 : {48159, 49666, 58224},                            # TIC
    4739  : {4802},                                           # Internode
    7470  : {17552, 38082, 132061},                           # TrueCorp
    8339  : {8559},                                           # Kabelplus
    19437 : {20454},                                          # Secured Servers
    9557  : {17557, 45595},                                   # Pakistan Telecommunication
    17425 : {133193},                                         # PHEA Thailand
    5650  : {7011},                                           # Frontier
    14618 : {16509, 38895},                                   # Amazon
    8447  : {15994},                                          # A1
    15169 : {19527},                                          # Google
    714   : {6185},                                           # Apple
    4230  : {28573},                                          # Embratel
    4826  : {9790},                                           # Vocus
    4657  : {10091, 38861},                                   # Starhub
    3786  : {17858},                                          # LG
    49889 : {50581},                                          # Ukrainian Telecom
    4134  : {4812, 23650, 134420, 136190},                    # Chinanet
    3216  : {8402},                                           # Vimpelcom
    4808  : {4837, 17621},                                    # China Unicom
    9808  : {24445},                                          # China Mobile
    6128  : {54004},                                          # Cablevision
    9063  : {8937},                                           # VSENET
    10620 : {14080},                                          # Telmex
    20459 : {36996},                                          # Telecom Namibia
    9049  : {12772},                                          # ER-Telecom
    3462  : {9680},                                           # Hinet
    12989 : {33438},                                          # Highwinds
    50719 : {60587},                                          # Xinon
    13285 : {9105},                                           # TalkTalk
    23951 : {58500},                                          # Citra
    8339  : {8559},                                           # Kabelplus
    15557 : {21502},                                          # SFR
    45430 : {45458, 133481},                                  # AdvancedWireless
    23520 : {27696},                                          # Columbus
    4775  : {132199},                                         # Globecom
    7713  : {17974},                                          # Telekom Indonesia
}

TIER1_SIBLINGS: t.Dict[int, t.Set[int]] = {
    t1: SIBLINGS[t1] if t1 in SIBLINGS else set() for t1 in TIER1
}

TIER1_SIBLINGS_FLAT: t.Set[int] = TIER1.union(
    reduce(set.union, TIER1_SIBLINGS.values())
)

def t1_rels() -> Source2Sink2Rel:
    out: Source2Sink2Rel = defaultdict(dict)
    for t1, siblings in TIER1_SIBLINGS.items():
        for sibling in siblings:
            out[t1][sibling] = 'p2p'
            out[sibling][t1] = 'p2p'
    for t1_1, t1_2 in permutations(TIER1_SIBLINGS.keys(), 2):
        out[t1_1][t1_2] = 'p2p'
        out[t1_2][t1_1] = 'p2p'
    return out