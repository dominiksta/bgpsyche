import unittest

from bgpsyche.stage2_enrich.geographic_distance import (
    geographic_distance_diff, _path_alternatives
)

class TestGeoGraphicDistance(unittest.TestCase):

    # def test_geographic_distance_diff(self):
    #     self.assertEquals(
    #         geographic_distance_diff([3320, 3320]),
    #         0
    #     )

    def test_path_alternatives(self):
        self.assertEquals(
            _path_alternatives([['US'], ['US']]),
            [['US', 'US']],
        )

        self.assertEquals(
            _path_alternatives([['US', 'DE'], ['US']]),
            [['US', 'US'], ['DE', 'US']]
        )

        self.assertEquals(
            _path_alternatives([['US', 'DE'], ['US', 'DE']]),
            [['US', 'US'], ['US', 'DE'], ['DE', 'US'], ['DE', 'DE']]
        )

        self.assertEquals(
            _path_alternatives([['US', 'DE'], ['US', 'DE', 'FR']]),
            [
                ['US', 'US'], ['US', 'DE'], ['US', 'FR'],
                ['DE', 'US'], ['DE', 'DE'], ['DE', 'FR']
            ]
        )


        # [['US', 'NL'], ['US'], ['US'], [], ['US']]
