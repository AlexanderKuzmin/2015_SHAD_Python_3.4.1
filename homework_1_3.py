'''
    @version: 1.1
    @since: 21.03.2015
    @author: Alexander Kuzmin
    @return: None
    @note: print a table with frequencies of letters in sorted order
'''

from sys import stdin
from operator import itemgetter
from collections import Counter

__author__ = 'Alexander Kuzmin'

def getSortedTableWithFrequencies(text):
    letters_counter = Counter()
    for word in text:
        lower_word = word.lower()
        if lower_word.isalpha():
            letters_counter[lower_word] += 1
    return sorted(sorted(letters_counter.items(), key=itemgetter(0)),
                    key=itemgetter(1), reverse=True)

if __name__ == '__main__':
    for line in getSortedTableWithFrequencies(stdin.read()):
        print("{0}: {1}".format(line[0], line[1]))
