'''
    @version: 1.0
    @since: 12.02.2015
    @author: Alexander Kuzmin
    @return: The list of the simple multiplies of the number
    @note:
'''

__author__ = 'Alexander Kuzmin'

def simple_multiplies(number):
    temp = number
    answer = {}
    for i in range(2, int(number**0.5)):
        degree = 0
        while number % i == 0:
            degree += 1
            number /= i
        if degree > 0:
            answer[i] = degree
        if number == 1:
            return answer

    if number % temp == 0:
        answer[temp] = 1
    return answer


if __name__ == '__main__':
    output = simple_multiplies(int(input()))
    print (output)
