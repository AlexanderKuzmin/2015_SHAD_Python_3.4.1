'''
    @version: 1.0
    @since: 07.04.2015
    @author: Alexander Kuzmin
    @note: Tools for processing a text: tokenizing, writing probabilities, generating and testing.
'''

import argparse
#import enum  # there are no the module in the contest
import random
import sys
from collections import Counter, defaultdict
from copy import copy
from itertools import chain
from operator import itemgetter

__author__ = 'Alexander Kuzmin'

def BinSearch(arr, predicate):
    '''
    Binary search by predicate.

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

def ToProcessText(input_text, args):
    '''
    Processor of the input_text.

    :param input_text: list of texts - the texts to be needed to process.
    :param args: argparse.ArgumentParser() - the command to be processed and the arguments for it.

    :return: result of command process.
    '''

    if (args.command == 'tokenize'):
        if (len(input_text) > 0):
            return Tokenize(input_text[0])
    elif (args.command == 'probabilities'):
        return GetProbabilities(input_text, args.depth)
    elif (args.command == 'generate'):
        return Generate(input_text, args.depth, args.size)
    elif (args.command == 'test'):
        return UnitTests(input_text, args)


# there are no module enum.Enum() in the contest
class State:
    alpha = 1
    digit = 2
    punctuation = 3
    space = 4

def Tokenize(text):
    '''
    Tokenize the current text.

    :param text: str - the text to be needed to tokenize.

    :return: list of tokens.
    '''

    tokens = []
    # there are no module enum in the contest
    # state = enum.Enum('state', 'alpha digit space punctuation')
    current_token = []
    current_state = -1
    for letter in text:
        if letter.isalpha():
            if current_state == State.alpha:
                current_token.append(letter)
            else:
                tokens.append("".join(current_token))
                current_state = State.alpha
                current_token = [letter]
        elif letter.isdigit():
            if current_state == State.digit:
                current_token.append(letter)
            else:
                tokens.append("".join(current_token))
                current_state = State.digit
                current_token = [letter]
        elif letter.isspace():
            if current_state == State.space:
                current_token.append(letter)
            else:
                tokens.append("".join(current_token))
                current_state = State.space
                current_token = [letter]
        else:
            if current_state == State.punctuation:
                current_token.append(letter)
            else:
                tokens.append("".join(current_token))
                current_state = State.punctuation
                current_token = [letter]
    tokens.append("".join(current_token))
    return tokens[1:]

def GetNGrams(tokens, n, predicate, tokens_counter=defaultdict(Counter)):
    '''
    put words n_grams from the list of different tokens into tokens_counter

    :param tokens: list fo str - the list of tokens
    :param n: int - the amount of words in one n-gram
    :param predicate: object - determines the correct predicates: predicate(token) should be true
    :param tokens_counter: collections.defaultdict - buffer for n-grams (may be not empty!)

    :return: defaultdict = collections.defaultdict of collections.Counter - dict of n-grams:
        'd-gram[:-1]': ('d-gram[-1]' : count)
    '''

    n_gram = []
    index = 0
    while (len(n_gram) != n + 1 and index < len(tokens)):
        if predicate(tokens[index]):
            n_gram.append(tokens[index])
        index += 1

    tokens_counter[tuple(n_gram[:-1])][n_gram[-1]] += 1

    while (index < len(tokens)):
        if predicate(tokens[index]):
            n_gram.pop(0)
            n_gram.append(tokens[index])
            tokens_counter[tuple(n_gram[:-1])][n_gram[-1]] += 1
        index += 1
    return tokens_counter

def GetProbabilities(input_text, depth):
    '''
    Get Probabilities of the text chains.

    :param input_text: list of str - the texts to be needed to tokenize.
    :param depth: int - the depth of the chain.

    :return: list of defauldict of Counters:
        'd-gram[:-1]': ('d-gram[-1]' : probability) for each d in 0..depth
    '''

    list_of_probabilities_counters = []
    tokens_lists = [Tokenize(text) for text in input_text]
    for current_depth in range(depth + 1):
        tokens_counter = defaultdict(Counter)
        for tokens in tokens_lists:
            tokens_counter = GetNGrams(tokens,
                                       current_depth,
                                       lambda x: x.isalpha(),
                                       tokens_counter)
        total_count = {k: sum(v.values()) for k, v in tokens_counter.items()}
        list_of_probabilities_counters.append(
            {prefix: {k: v / total_count[prefix] for k, v in counter.items()}
                for prefix, counter in tokens_counter.items()})
    return list_of_probabilities_counters

def Generate(text, depth, size, seed=None):
    '''
    generate new text according with probabilities of depth-grams from the text

    :param text: str - text, from which we get distribution of d-grams
    :param depth: int - the depth (length) of d-grams
    :param size: int - the length of the text
    :param seed: int - random seed for random.seed()

    :return: generated text
    '''

    n_gram_probabilities = GetNGrams(Tokenize(text),
                                     depth,
                                     lambda x: not x.isspace() and not x.isdigit())
    total_arrays = {key: list(chain(*[[k] * v for k, v in counter.items()]))
                    for key, counter in n_gram_probabilities.items()}
    prefices = list([word for word in n_gram_probabilities.keys()
                     if len(word) == 0 or word[0][0].isupper()])
    generated_text = []
    prefix_tokens = []
    if (prefices != []):
        prefix_tokens = list(random.choice(prefices))
        generated_text = copy(prefix_tokens)
    random.seed(seed)
    current_size = sum([len(word) for word in generated_text])
    while(current_size < size):
        current_array = total_arrays.get(tuple(prefix_tokens))
        if (current_array is not None):
            word = random.choice(current_array)
            generated_text.append(word)
            current_size += len(word) + 1
            prefix_tokens.append(word)
            prefix_tokens.pop(0)
        else:
            prefix_tokens = list(random.choice(list(n_gram_probabilities.keys())))
            generated_text.extend(prefix_tokens)
            current_size += sum([len(word) for word in prefix_tokens]) + 1
    return " ".join(generated_text)

def ReadInputText(args):
    input_text = []
    if args.command == 'tokenize':
        input_text.append(input())
    else:
        for text in sys.stdin:
            input_text.append(text)
    return input_text

def Print(text, args):
    '''
    print the text in format according with args

    :param text: str - the text to be printed.
    :param args: argparse.ArgumentParser() - the command to be processed and the arguments for it.

    :return: void
    '''

    if (args.command == 'tokenize'):
        for word in text:
            print(word)
    elif (args.command == 'generate'):
        print(text)
    elif (args.command == 'probabilities'):
        for depth_dict in text:
            sorted_keys = sorted(depth_dict.keys())
            for prefix in sorted_keys:
                print(" ".join(prefix))
                sorted_dict = sorted(depth_dict[prefix].items(), key=itemgetter(0))
                for word, count in sorted_dict:
                    print("  {0}: {1:.2f}".format(word, count))
    elif (args.command == 'test'):
        None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = ("Tools for processing a text: tokenizing, writing probabilities, "
                          "generating and testing.")
    parser.add_argument("command", type=str,
                        help="command to process")
    parser.add_argument("-d", "--depth", action="store", type=int, default=1,
                        help="The maximum depth of chains.")
    parser.add_argument("-s", "--size", action="store", type=int, default=32,
                        help="Approximate amount of words for generating.")

    args_raw_str = input()
    args = parser.parse_args(args_raw_str.split())
    input_text = ReadInputText(args)
    Print(ToProcessText(input_text, args), args)

def UnitTests(input_text, args):
    '''
    Unit tests for the program

    :param input_text:
    :param args:

    :return:
    '''
