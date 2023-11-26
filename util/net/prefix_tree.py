import typing as t
from ipaddress import ip_address, ip_network

from util.net import IPAddress, IPNetwork

def _ip2bin(ip: IPAddress) -> str: return '{:032b}'.format(int(ip))
# def _bin2ip(bin: str) -> IPAddress: return ip_address(int(bin, 2))

_T = t.TypeVar('_T')
_HashableT = t.TypeVar('_HashableT', bound=t.Hashable)

class _PrefixTreeNode(t.Generic[_HashableT, _T]):
    def __init__(
            self,
            key: _HashableT,
            value: t.Optional[_T] = None,
    ) -> None:
        self.key = key
        self.value = value
        self.__children: t.Dict[_HashableT, '_PrefixTreeNode'] = {}

    def add_child(self, child: '_PrefixTreeNode'):
        assert not child.key in self.__children
        self.__children[child.key] = child

    def get_child(self, key: _HashableT) -> '_PrefixTreeNode':
        assert key in self.__children
        return self.__children[key]

    def has_child(self, key: _HashableT) -> bool:
        return key in self.__children


class NetworkPrefixTree:
    """A datastructure to search for an IP in a set of prefixes.

    This is essentially a "Patricia Trie" or "Prefix Tree".

    The implementation is not exactly on the level of an optimized C program
    running on a real router, but its good enough for now.

    [2023-10-17:dominiksta]: I could not figure out how this could generalize to
    a "normal" prefix tree, so this works slightly differently.
    """

    def __init__(self) -> None:
        self.root = _PrefixTreeNode('')
        self.__count = 0

    def insert(self, net: IPNetwork):
        self.__count += 1
        addr = _ip2bin(net.network_address)

        assert len(addr) == 32, net
        node = self.root
        for i in range(net.prefixlen):
            char = addr[i]
            if not node.has_child(char):
                node.add_child(_PrefixTreeNode(char))
            node = node.get_child(char)

        node.value = net


    def search(self, addr: IPAddress) -> t.Optional[IPNetwork]:
        addr_str = _ip2bin(addr)

        node = self.root
        for i in range(len(addr_str)):
            char = addr_str[i]
            if node.value is not None: return node.value
            if not node.has_child(char): return None
            node = node.get_child(char)


    def __repr__(self) -> str:
        return f'<NetworkPrefixTree size={self.__count}>'


    @staticmethod
    def from_list(nets: t.Iterable[IPNetwork]) -> 'NetworkPrefixTree':
        t = NetworkPrefixTree()
        for net in nets: t.insert(net)
        return t


def _test():

    tree = NetworkPrefixTree()

    net1 = ip_network('192.168.178.0/24')
    tree.insert(net1)
    assert tree.search(ip_address('192.168.178.0')) == net1
    assert tree.search(ip_address('192.168.178.20')) == net1
    assert tree.search(ip_address('192.168.177.0')) is None

    net2 = ip_network('192.168.178.0/28')
    tree.insert(net2)
    assert tree.search(ip_address('192.168.178.0')) == net1

    net3 = ip_network('192.168.0.0/22')
    tree.insert(net3)
    assert tree.search(ip_address('192.168.3.0')) == net3
    assert tree.search(ip_address('192.168.4.0')) is None
    assert tree.search(ip_address('192.168.178.0')) == net1

    print(tree)



if __name__ == '__main__': _test()