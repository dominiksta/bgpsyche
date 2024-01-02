from pprint import pformat
import typing as t
from ipaddress import (
    IPv4Address, IPv4Network,
    IPv6Address, IPv6Network,
)

from bgpsyche.util.net import IPAddress, IPNetwork

def _ip2bin(ip: IPAddress) -> str:
    if ip.version == 4: return '{:032b}'.format(int(ip))
    else: return '{:0128b}'.format(int(ip))
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
        assert key in self.__children, self.__children
        return self.__children[key]

    def has_child(self, key: _HashableT) -> bool:
        return key in self.__children


class PrefixTree:
    def __init__(self):
        self.root = _PrefixTreeNode(0)
        self.__paths: t.Set[str] = set()

    def insert(self, path: t.Iterable[int]):
        path = tuple(path)
        path_str = ','.join(map(str, path))
        if path_str in self.__paths: return
        self.__paths.add(path_str)

        current = self.root
        for i, char in enumerate(path):
            if not current.has_child(char):
                # print({ 'char': char, 'prefix': path[0:i+1] })
                current.add_child(_PrefixTreeNode(char))
            current = current.get_child(char)

    def search(self, path: t.Iterable[int]) -> bool:
        path = tuple(path)
        current = self.root
        for char in path:
            if not current.has_child(char):
                return False
            current = current.get_child(char)
        return True

    @staticmethod
    def from_iter(paths: t.Iterable[t.Iterable[int]]) -> 'PrefixTree':
        tree = PrefixTree()
        for path in paths: tree.insert(path)
        return tree

    @property
    def paths(self): return self.__paths



class _NetworkPrefixTree:
    """A datastructure to search for an IP in a set of prefixes.

    This is essentially a "Patricia Trie" or "Prefix Tree".

    The implementation is not exactly on the level of an optimized C program
    running on a real router, but its good enough for now.

    [2023-10-17:dominiksta]: I could not figure out how this could generalize to
    a "normal" prefix tree, so this works slightly differently.
    """

    def __init__(self) -> None:
        self.root = _PrefixTreeNode('')
        self.__networks: t.Set[IPNetwork] = set()


    def insert(self, net: IPNetwork):
        if net in self.__networks: return
        self.__networks.add(net)
        addr = _ip2bin(net.network_address)

        assert len(addr) == 32 or len(addr) == 128, pformat({
            'net': net, 'bin': addr
        })
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


    @property
    def networks(self) -> t.Set[IPNetwork]:
        return self.__networks


    def __repr__(self) -> str:
        return f'<{self.__class__} size={len(self.__networks)}>'


    @staticmethod
    def from_list(nets: t.Iterable[IPNetwork]) -> '_NetworkPrefixTree':
        t = _NetworkPrefixTree()
        for net in nets: t.insert(net)
        return t


class NetworkPrefixTreeV4(_NetworkPrefixTree):
    def insert(self, net: IPv4Network): return super().insert(net)
    def search(self, addr: IPv4Address) -> t.Optional[IPv4Network]:
        return t.cast(IPv4Network, super().search(addr))

    @staticmethod
    def from_list(nets: t.Iterable[IPv4Network]) -> 'NetworkPrefixTreeV4':
        t = NetworkPrefixTreeV4()
        for net in nets: t.insert(net)
        return t


class NetworkPrefixTreeV6(_NetworkPrefixTree):
    def insert(self, net: IPv6Network): return super().insert(net)
    def search(self, addr: IPv6Address) -> t.Optional[IPv6Network]:
        return t.cast(IPv6Network, super().search(addr))

    @staticmethod
    def from_list(nets: t.Iterable[IPv6Network]) -> 'NetworkPrefixTreeV6':
        t = NetworkPrefixTreeV6()
        for net in nets: t.insert(net)
        return t


def make_prefix_trees_for_list(
        prefixes: t.Iterable[IPNetwork]
) -> t.Tuple[NetworkPrefixTreeV4, NetworkPrefixTreeV6]:
    tree_v4 = NetworkPrefixTreeV4()
    tree_v6 = NetworkPrefixTreeV6()
    for prefix in prefixes:
        if prefix.version == 4: tree_v4.insert(prefix)
        else: tree_v6.insert(prefix)

    return tree_v4, tree_v6


def _test_basic():
    tree = PrefixTree()
    tree.insert([1, 2, 3])
    tree.insert([1, 2, 3, 4])
    tree.insert([1, 2, 3, 4, 5])
    tree.insert([1, 2, 4])

    assert     tree.search([1, 2])
    assert not tree.search([1, 2, 5])
    assert     tree.search([1, 2, 4])
    assert     tree.search([1, 2, 3, 4, 5])
    assert not tree.search([2, 3, 4])
    assert not tree.search([2, 3])


def _test():

    tree = NetworkPrefixTreeV4()

    net1 = IPv4Network('192.168.178.0/24')
    tree.insert(net1)
    assert tree.search(IPv4Address('192.168.178.0')) == net1
    assert tree.search(IPv4Address('192.168.178.20')) == net1
    assert tree.search(IPv4Address('192.168.177.0')) is None

    net2 = IPv4Network('192.168.178.0/28')
    tree.insert(net2)
    assert tree.search(IPv4Address('192.168.178.0')) == net1

    net3 = IPv4Network('192.168.0.0/22')
    tree.insert(net3)
    assert tree.search(IPv4Address('192.168.3.0')) == net3
    assert tree.search(IPv4Address('192.168.4.0')) is None
    assert tree.search(IPv4Address('192.168.178.0')) == net1

    print(tree)



if __name__ == '__main__':
    _test()
    _test_basic()