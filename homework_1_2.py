'''
    @version: 1.1
    @since: 21.03.2015
    @author: Alexander Kuzmin
    @return: a non-negative float number
    @note: L_p norm of a vector
'''

__author__ = 'Alexander Kuzmin'

def pNorm(vector, p):
    list_of_coordinates = [abs(float(x)) ** p for x in vector]
    return sum(list_of_coordinates) ** (1 / p)

if __name__ == '__main__':
    p = float(input())
    print(pNorm(input().strip().split(), p))
