'''
    @version: 1.0
    @since: 07.04.2015
    @author: Alexander Kuzmin
    @return:
    @note: for parsing of arguments
'''

import argparse

__author__ = 'Alexander Kuzmin'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = ("The square of the number.")
    parser.add_argument("number", type=int,
                        help="display a square of a given number")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    args = parser.parse_args()
    answer = args.square**2
    if args.verbose:
        print("the square of {} equals {}".format(args.square, answer))
    else:
        print(answer)