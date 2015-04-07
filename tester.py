'''
    @version: 1.0
    @since: 02.03.2015
    @author: Alexander Kuzmin
    @return:
    @note: for tests of commands
'''

import unittest

__author__ = 'Alexander Kuzmin'

class TestProgram(unittest.TestCase):

    def test_01(self):
        self.assertTrue([1] == [] + [1], "Bad message 2")

    def test_02(self):
        self.assertEqual(1, 1, "Bad message 1")

if __name__ == '__main__':
    unittest.main()
