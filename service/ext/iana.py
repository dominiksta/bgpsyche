import typing as t
import xml.etree.ElementTree

from util.const import DATA_DIR
from util.net import download_file_cached

_iana_invalid_ranges: t.List[t.Tuple[int, int]] = []
def asn_is_iana_assigned(asn: int) -> bool:
    global _iana_invalid_ranges

    if len(_iana_invalid_ranges) == 0:
        invalid_keywords = [
            'reserved',
            'private use',
            'documentation and sample code',
            'unallocated',
        ]

        def is_invalid(desc: str) -> bool:
            lower = desc.lower()
            for keyword in invalid_keywords:
                if keyword in lower: return True
            return False

        file = download_file_cached(
            'http://www.iana.org/assignments/as-numbers/as-numbers.xml',
            DATA_DIR / 'iana-as-numbers.xml'
        )
        ns = { 'iana_assign': 'http://www.iana.org/assignments' }
        xml_parsed = xml.etree.ElementTree.parse(file).getroot()

        # print(xml_parsed.findall('iana_assign:registry', ns))
        records = [
            registry.findall('iana_assign:record', ns) for registry in
            xml_parsed.findall('iana_assign:registry', ns)
        ]
        # print(records)
        # flatten
        records = [item for sublist in records for item in sublist]
        for r in records:
            if not is_invalid(str(r.find('iana_assign:description', ns).text)):
                continue
            # print(r)
            pair = tuple(map(int, r.find('iana_assign:number', ns).text.split('-')))
            if len(pair) == 1: pair = (pair[0], pair[0])
            _iana_invalid_ranges.append(pair)
        # print(_iana_invalid_ranges)

    for _range in _iana_invalid_ranges:
        # print(_range)
        if _range[0] <= asn and _range[1] >= asn: return False
    return True
