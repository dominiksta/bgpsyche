import unittest

from bgpsyche.util.bgp.path_prepending import eliminate_path_prepending

class TestPathPrepending(unittest.TestCase):

    def test_eliminate_path_prepending(self):
        self.assertEquals(
            eliminate_path_prepending([1, 2, 3, 4]),
            [1, 2, 3, 4]
        )

        self.assertEquals(
            eliminate_path_prepending([1, 1, 2, 3, 4]),
            [1, 2, 3, 4]
        )

        self.assertEquals(
            eliminate_path_prepending([1, 2, 2, 3, 4]),
            [1, 2, 3, 4]
        )

        self.assertEquals(
            eliminate_path_prepending([1, 2, 3, 4, 4]),
            [1, 2, 3, 4]
        )

        self.assertEquals(
            eliminate_path_prepending([1, 2, 2, 3, 4, 4]),
            [1, 2, 3, 4]
        )

        self.assertEquals(
            eliminate_path_prepending([1, 1, 2, 2, 3, 3, 4, 4, 4, 4]),
            [1, 2, 3, 4]
        )