# Credits to https://stackoverflow.com/a/73155753

import glob
import os
import unittest


def test_suite_from_recursive_discover(pattern):
    test_files = glob.glob('**/{}'.format(pattern), recursive=True)
    test_dirs = list(set(([
        os.path.dirname(os.path.abspath(test_file))
        for test_file in test_files
    ])))
    suites = [
        unittest.TestLoader().discover(start_dir=d, pattern=pattern)
        for d in test_dirs
    ]
    suite = unittest.TestSuite(suites)
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite_from_recursive_discover('*_test.py'))
