'''
    @version: 1.0
    @since: 07.04.2015
    @author: Alexander Kuzmin
    @note: Tools for processing a text: tokenizing, writing probabilities, generating and testing.
'''

import argparse
import enum
import random
from collections import Counter
from collections import defaultdict
from copy import copy
from itertools import chain

__author__ = 'Alexander Kuzmin'

'''
    Questions:
        1. open file or read from input()? - input 
        2. How to tokenize "z1?af2back, toto"? OK
        3. All tokens (Punctuation, not spaces) in probabilities?
            3.1 Ignore 2-grams "..., ...", "..., man..." or "...! I..."? only words
        4. How to process 'What's that...?'?  OK
        5. Слова, для которых вероятность равна 0, выводить не нужно. OK
        6. Обратите внимание, что первым всегда идет блок,
            соответствующий глубине 0 и он обозначается пустой строке - \n + '  :'...
        7. Какая сложность генерации допустима? any
        8. Any balanced tries? no
'''

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

def ToProcessText(text, args):
    '''
    Processor of the text.

    :param text: str - the text to be needed to process.
    :param args: argparse.ArgumentParser() - the command to be processed and the arguments for it.

    :return: result of command process.
    '''

    if (args.command == 'tokenize'):
        return Tokenize(text)
    elif (args.command == 'probabilities'):
        return GetProbabilities(text, args.depth)
    elif (args.command == 'generate'):
        return Generate(text, args.depth, args.size)

def Tokenize(text):
    '''
    Tokenize the current text.

    :param text: str - the text to be needed to tokenize.

    :return: list of tokens.
    '''

    tokens = []
    state = enum.Enum('state', 'alpha digit space punctuation')
    current_token = []
    current_state = -1
    for letter in text:
        if letter.isalpha():
            if current_state == state.alpha:
                current_token.append(letter)
            else:
                tokens.append("".join(current_token))
                current_state = state.alpha
                current_token = [letter]
        elif letter.isdigit():
            if current_state == state.digit:
                current_token.append(letter)
            else:
                tokens.append("".join(current_token))
                current_state = state.digit
                current_token = [letter]
        elif letter.isspace():
            if current_state == state.space:
                current_token.append(letter)
            else:
                tokens.append("".join(current_token))
                current_state = state.space
                current_token = [letter]
        else:
            if current_state == state.punctuation:
                current_token.append(letter)
            else:
                tokens.append("".join(current_token))
                current_state = state.punctuation
                current_token = [letter]
    tokens.append("".join(current_token))
    return tokens[1:]

def GetNGrams(tokens, n):
    '''
    get words n_grams from the list of different tokens

    :param tokens: list fo str - the list of tokens
    :param n: int - the amount of words in one n-gram

    :return: collections.defaultdict of collections.Counter - dict of n-grams
    '''

    n_grams = defaultdict(Counter)
    n_gram = []
    index = 0
    while (len(n_gram) != n + 1 and index < len(tokens)):
        if tokens[index].isalpha():
            n_gram.append(tokens[index])
        index += 1

    n_grams[tuple(n_gram[:-1])][n_gram[-1]] += 1

    while (index < len(tokens)):
        if tokens[index].isalpha():
            n_gram.pop(0)
            n_gram.append(tokens[index])
            n_grams[tuple(n_gram[:-1])][n_gram[-1]] += 1
        index += 1
    return n_grams

def GetProbabilities(text, depth):
    '''
    Get Probabilities of the text chains.

    :param text: str - the text to be needed to tokenize.
    :param depth: int - the depth of the chain.

    :return: list of defauldict of Counters:
    'd-gram[:-1]': ('d-gram[-1]' : probability) for each d in 0..depth
    '''

    list_of_probabilities_counters = []
    tokens = Tokenize(text)
    for current_depth in range(depth + 1):
        tokens_counter = GetNGrams(tokens, current_depth)
        total_count = {k : sum(v.values()) for k, v in tokens_counter.items()}
        list_of_probabilities_counters.append(
            {prefix : {k : v / total_count[prefix] for k, v in counter.items()}
                for prefix, counter in tokens_counter.items()})
    return list_of_probabilities_counters

def Generate(text, depth, size, seed=123):
    '''
    generate new text according with probabilities of depth-grams from the text

    :param text: str - text, from which we get distribution of d-grams
    :param depth: int - the depth (length) of d-grams
    :param size: int - the length of the text
    :param seed: int - random seed for random.seed()

    :return: generated text
    '''

    n_gram_probabilities = GetNGrams(text, depth)
    total_arrays = {key : list(chain([[k] * v for k, v in counter.items()]))
                    for key, counter in n_gram_probabilities.items()}
    print(total_arrays)
    prefix_tokens = list(random.choice(list(n_gram_probabilities.keys())))
    print(prefix_tokens)
    generated_text = copy(prefix_tokens)
    random.seed(seed)
    while(len(generated_text) < size):
        value = random.choice(total_arrays[tuple(prefix_tokens)])
        generated_text.append(value)
        prefix_tokens.append(value)
        prefix_tokens.pop(0)
    # print(generated_text)
    return " ".join(generated_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = ("Tools for processing a text: tokenizing, writing probabilities, "
                          "generating and testing.")
    parser.add_argument("command", type=str,
                        help="command to process")
    parser.add_argument("-d", "--depth", action="store", type=int, default=0,
                        help="The maximum depth of chains.")
    parser.add_argument("-s", "--size", action="store", type=int, default=32,
                        help="Approximate amount of words for generating.")
    args = parser.parse_args()
    text = input()
    while text:
        print(ToProcessText(text, args))
        text = input()
