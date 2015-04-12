'''
    @version: 1.0
    @since: 07.04.2015
    @author: Alexander Kuzmin
    @note: Tools for processing a text: tokenizing, writing probabilities, generating and testing.
'''

#import enum  # there are no such module in the contest
import argparse
import unittest
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
        return UnitTests()

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

def Generate(input_text, depth, size, seed=None):
    '''
    generate new text according with probabilities of depth-grams from the text

    :param text: list of str - text, from which we get distribution of d-grams
    :param depth: int - the depth (length) of d-grams
    :param size: int - the length of the text
    :param seed: int - random seed for random.seed()

    :return: generated text
    '''

    random.seed(seed)
    n_gram_probabilities = defaultdict(Counter)
    for line in input_text:
        n_gram_probabilities = GetNGrams(Tokenize(line),
                                         depth,
                                         lambda x: not x.isspace() and not x.isdigit(),
                                         n_gram_probabilities)
    total_arrays = {key: list(chain(*[[k] * v for k, v in counter.items()]))
                    for key, counter in n_gram_probabilities.items()}
    prefices = [word for word in n_gram_probabilities.keys()
                if len(word) == 0 or word[0][0].isupper()]
    generated_text = []
    prefix_tokens = []
    if (prefices != []):
        prefix_tokens = list(random.choice(prefices))
        generated_text = copy(prefix_tokens)
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

class TestProgram(unittest.TestCase):
    '''
    Unit tests for the program
    '''

    def test_Tokenize_01(self):
        self.assertListEqual(Tokenize("Hello, world!"),
                             ["Hello", ",", " ", "world", "!"],
                             "Tokenize test failed.")

    def test_Tokenize_02(self):
        self.assertListEqual(Tokenize("Abracadabra"), ["Abracadabra"], "Tokenize test failed.")

    def test_Tokenize_03(self):
        self.assertListEqual(Tokenize("18572305232"), ["18572305232"], "Tokenize test failed.")

    def test_Tokenize_04(self):
        self.assertListEqual(Tokenize(",!...%^&*@()!@"),
                             [",!...%^&*@()!@"],
                             "Tokenize test failed.")

    def test_Tokenize_04(self):
        self.assertListEqual(Tokenize("         "), ["         "], "Tokenize test failed.")

    def test_Tokenize_05(self):
        self.assertListEqual(
            Tokenize("What's that? I don't know..."),
            ["What", "'", "s", " ", "that", "?", " ",
             "I", " ", "don", "'", "t", " ", "know", "..."],
            "Tokenize test failed.")

    def test_Tokenize_06(self):
        self.assertListEqual(Tokenize("2kj2h5^fa$s.64f"),
                             ["2", "kj", "2", "h", "5", "^", "fa", "$", "s", ".", "64", "f"],
                             "Tokenize test failed.")

    def test_Tokenize_07(self):
        self.assertListEqual(Tokenize(" "),
                             [" "],
                             "Tokenize test failed.")


    def test_Probabilities_01(self):
        self.assertListEqual(
            GetProbabilities(["First test string", "Second test line"], 1),
            [{(): {'First': 0.16666666666666666,
                   'line': 0.16666666666666666,
                   'string': 0.16666666666666666,
                   'test': 0.3333333333333333,
                   'Second': 0.16666666666666666}},
             {('First',): {'test': 1.0},
              ('test',): {'line': 0.5, 'string': 0.5},
              ('Second',): {'test': 1.0}}],
            "Probabilities test failed.")

    def test_Probabilities_02(self):
        self.assertListEqual(
            GetProbabilities(["a a a a a", "b b b b b b"], 1),
            [{(): {'a': 0.45454545454545453, 'b': 0.5454545454545454}},
             {('a',): {'a': 1.0}, ('b',): {'b': 1.0}}],
            "Probabilities test failed.")

    def test_Probabilities_03(self):
        self.assertListEqual(
            GetProbabilities(["a b c d e f g"], 4),
            [{(): {'a': 0.14285714285714285,
                   'b': 0.14285714285714285,
                   'c': 0.14285714285714285,
                   'f': 0.14285714285714285,
                   'e': 0.14285714285714285,
                   'd': 0.14285714285714285,
                   'g': 0.14285714285714285}},
             {('a',): {'b': 1.0},
              ('b',): {'c': 1.0},
              ('c',): {'d': 1.0},
              ('d',): {'e': 1.0},
              ('e',): {'f': 1.0},
              ('f',): {'g': 1.0}},
             {('a', 'b'): {'c': 1.0},
              ('b', 'c'): {'d': 1.0},
              ('c', 'd'): {'e': 1.0},
              ('d', 'e'): {'f': 1.0},
              ('e', 'f'): {'g': 1.0}},
             {('a', 'b', 'c'): {'d': 1.0},
              ('b', 'c', 'd'): {'e': 1.0},
              ('c', 'd', 'e'): {'f': 1.0},
              ('d', 'e', 'f'): {'g': 1.0}},
             {('a', 'b', 'c', 'd'): {'e': 1.0},
              ('b', 'c', 'd', 'e'): {'f': 1.0},
              ('c', 'd', 'e', 'f'): {'g': 1.0}}],
            "Probabilities test failed.")

    def test_Probabilities_04(self):
        self.assertListEqual(GetProbabilities(["a b c"], 10),
                             [{(): {'a': 0.3333333333333333,
                                    'b': 0.3333333333333333,
                                    'c': 0.3333333333333333}},
                              {('a',): {'b': 1.0},
                               ('b',): {'c': 1.0}},
                              {('a', 'b'): {'c': 1.0}},
                              {('a', 'b'): {'c': 1.0}},
                              {('a', 'b'): {'c': 1.0}},
                              {('a', 'b'): {'c': 1.0}},
                              {('a', 'b'): {'c': 1.0}},
                              {('a', 'b'): {'c': 1.0}},
                              {('a', 'b'): {'c': 1.0}},
                              {('a', 'b'): {'c': 1.0}},
                              {('a', 'b'): {'c': 1.0}}],
                             "Probabilities test failed.")

    def test_Probabilities_05(self):
        self.assertListEqual(GetProbabilities(["Hello, world!"], 1),
                             [{(): {'world': 0.5, 'Hello': 0.5}},
                              {('Hello',): {'world': 1.0}}],
                             "Probabilities test failed.")

    def test_Generate_01(self):
        self.assertLess(
            len(Generate(["Python - активно развивающийся язык программирования, новые версии "
                          "(с добавлением/изменением языковых свойств) выходят примерно раз в два"
                          " с половиной года.",
                          "Вследствие этого и некоторых других причин на Python отсутствуют "
                          "стандарт ANSI, ISO или другие официальные стандарты, их роль "
                          "выполняет CPython."],
                         1,
                         10,
                         seed=321)),
            30,
            "Generate test failed.")

    def test_Generate_02(self):
        self.assertGreater(
            len(Generate(["Python - активно развивающийся язык программирования, новые версии "
                          "(с добавлением/изменением языковых свойств) выходят примерно раз в два"
                          " с половиной года.",
                          "Вследствие этого и некоторых других причин на Python отсутствуют "
                          "стандарт ANSI, ISO или другие официальные стандарты, их роль "
                          "выполняет CPython."],
                         10,
                         10,
                         seed=321)),
            40,
            "Generate test failed.")

def UnitTests():
    '''
    Unit tests for the program

    :return: None
    '''

    unittest.main()

def ReadInputText(args):
    '''
    read input data depending on args

    :param args: argparse.ArgumentParser() - the arguments of the program.

    :return: list of str - text from the input
    '''
    input_text = []
    if args.command == 'tokenize':
        input_text.append(input())
    elif (args.command == 'generate' or args.command == 'probabilities'):
        for text in sys.stdin:
            input_text.append(text)
    elif (args.command == 'test'):
        None
    return input_text

def Print(text, args):
    '''
    print the text in format according with args

    :param text: str - the text to be printed.
    :param args: argparse.ArgumentParser() - the command to be processed and the arguments for it.

    :return: None
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

    input_text = []
    with open("input.txt", encoding='utf-8') as reader:
        input_text = reader.read().split("\n")
    args = parser.parse_args(input_text[0].split())
    Print(ToProcessText(input_text[1:], args), args)
    # args = parser.parse_args(input().split())
    # input_text = ReadInputText(args)
    # Print(ToProcessText(input_text, args), args)
