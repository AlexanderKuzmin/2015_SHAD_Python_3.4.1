'''
    @version: 1.1
    @since: 21.03.2015
    @author: Alexander Kuzmin
    @return: a non-negative integer number
    @note: a fibonacci value
'''

__author__ = 'Alexander Kuzmin'

def fibonacci(n):
    answer = [0, 1]
    for idx in range(n):
        answer = [answer[1], sum(answer)]
    return answer[0]

if __name__ == '__main__':
    print(fibonacci(int(input())))
