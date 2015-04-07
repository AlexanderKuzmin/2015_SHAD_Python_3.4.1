'''
    @version: 1.0
    @since: 07.04.2015
    @author: Alexander Kuzmin
    @note: Tools for processing a text: tokenizing, writing probabilities, generating and testing.
'''

import argparse
import enum

__author__ = 'Alexander Kuzmin'

'''
    Questions:
        1. open file or read from input()?
        2. How to tokenize "z1?af2back, toto"?
'''

def ToProcessText(text, command):
    '''
    :param text: str - the text to be needed to process.
    :param command: str - the name of command to be used for process.
    :return: result of command process.
    '''
    if (command == 'tokenize'):
        return Tokenize(text)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument()


    text = input()
    result = ToProcessText(text, "tokenize")
    print(result)

    result = ToProcessText(text, "probabilities --depth 1")
