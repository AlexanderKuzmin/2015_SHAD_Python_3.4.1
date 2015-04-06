import unittest

__author__ = 'Alexander Kuzmin'
'''
    @version: 1.0
    @since: 02.03.2015
    @author: Alexander Kuzmin
    @return:
    @note: for tests of commands
'''

class TestProgram(unittest.TestCase):
    def test_01(self):
        self.assertEqual(1, 1, "Bad message 1")
    def test_02(self):
        self.assertTrue([1] == [] + [1], "Bad message 2")

if __name__ == '__main__':
    unittest.main()
