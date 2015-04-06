'''
    @version: 1.1
    @since: 21.03.2015
    @author: Alexander Kuzmin
    @return: text
    @note: breaks the text into lines (text with '\n') less than maxLength
'''

import sys

__author__ = 'Alexander Kuzmin'

def breaksTheTextIntoLines(text, maxLength):
    text = text.split(" ")
    textBuffer = []
    capacity = 0
    start = True
    for word in text:
        syllables = word.split("\n")  # if there were symbols '\n'
        # the first syllable
        if not start:
            if maxLength < capacity + len(syllables[0]) + 1:
                textBuffer.append("\n")
                capacity = 0
            else:
                textBuffer.append(" ")
                capacity += 1
        textBuffer.append(syllables[0])
        capacity += len(syllables[0])
        start = False
        # other syllables
        for idx in range(1, len(syllables)):
            textBuffer.append("\n")
            textBuffer.append(syllables[idx])
            capacity = len(syllables[idx])
    return "".join(textBuffer)

if __name__ == '__main__':
    maxLength = int(input())
    print(breaksTheTextIntoLines(sys.stdin.read(), maxLength))
