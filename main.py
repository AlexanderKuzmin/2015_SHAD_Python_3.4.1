'''
    @version: 1.0
    @since: 07.04.2015
    @author: Alexander Kuzmin
    @return:
    @note: for small tests
'''

from collections import OrderedDict
from collections import Counter
#import rbtree, RBTree, pyavl


def BinSearch(arr, predicate):
    '''
    Bin search by predicate.

    :param arr: list - where to search
    :param predicate: function - we looking for the element that satisfies the predicate

    :return: index of the first element that satisfies predicate(element) is True
    '''

    left, right = 0, len(arr) - 1
    while left <= right:
        mid = int(left + (right - left) / 2)
        if predicate(arr[mid]):
            if not predicate(arr[mid - 1]) or mid == 0:
                return mid
            right = mid - 1
        else:
            left = mid + 1
    return -1

if __name__ == '__main__':
    a = ()
    print(type(a))
    print(BinSearch(['a'], lambda x: x.isalpha()), 0)

    print(BinSearch(['.', 'a'], lambda x: x.isalpha()), 1)
    print(BinSearch(['.'], lambda x: x.isalpha()), -1)
    print(BinSearch(['a', 'a'], lambda x: x.isalpha()), 0)
    print(BinSearch([], lambda x: x.isalpha()), -1)
    print(BinSearch(['.', 'a', 'b'], lambda x: x.isalpha()), 1)
    print(BinSearch(['.', 'c', 'd', 'e'], lambda x: x.isalpha()), 1)
    print(BinSearch(['.', ',', 'a'], lambda x: x.isalpha()), 2)

    z = OrderedDict()
    z['1']