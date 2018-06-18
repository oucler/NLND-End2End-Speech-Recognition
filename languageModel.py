#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 10:39:12 2018

@author: ucleraiserver
"""
from textblob import TextBlob
#import utils
from collections import Counter, defaultdict
import numpy as np
import re
from data_generator import AudioGenerator
import numpy as np


def generate_corpus(desc_file):
    data_sentences = AudioGenerator()
    #data_gen.load_train_data(desc_file=desc_file)
    data_sentences.load_train_data(desc_file=desc_file)
    sentences = data_sentences.train_texts
    return sentences
def words(text): return re.findall(r'\w+', text.lower())



def readCorpus(sentences):
    words_ = []
    for sentece in sentences:
        for word in sentece.split():
            words_.append(word)
            #print (words)
    counter = Counter(words_)
    return counter
    #print(counter)
    #WORDS = Counter(words(file))
    #WORDS = Counter(words(open('big.txt').read()))

WORDS = Counter()
WORDS = readCorpus(generate_corpus(desc_file='train_clean_corpus.json'))

def P(word): 
    "Probability of `word`."
    N=sum(WORDS.values())
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))



def bigrams_from_transcript(filename):
    """
    read a file of sentences, adding start '<s>' and stop '</s>' tags; Tokenize it into a list of lower case words
    and bigrams
    :param filename: string 
        filename: path to a text file consisting of lines of non-puncuated text; assume one sentence per line
    :return: list, list
        tokens: ordered list of words found in the file
        bigrams: a list of ordered two-word tuples found in the file
    """
    tokens = []
    bigrams = []
    with open(filename, 'r') as f:
        for line in f:
            line_tokens, line_bigrams = sentence_to_bigrams(line)
            tokens = tokens + line_tokens
            bigrams = bigrams + line_bigrams
    return tokens, bigrams

def bigrams_from_sentences(sentences):
    """
    read a file of sentences, adding start '<s>' and stop '</s>' tags; Tokenize it into a list of lower case words
    and bigrams
    :param filename: string 
        filename: path to a text file consisting of lines of non-puncuated text; assume one sentence per line
    :return: list, list
        tokens: ordered list of words found in the file
        bigrams: a list of ordered two-word tuples found in the file
    """
    tokens = []
    bigrams = []
    for sentence in sentences:
        line_tokens, line_bigrams = sentence_to_bigrams(sentence)
        tokens = tokens + line_tokens
        bigrams = bigrams + line_bigrams
    return tokens, bigrams    
    
def sentence_to_bigrams(sentence):
    """
    Add start '<s>' and stop '</s>' tags to the sentence and tokenize it into a list
    of lower-case words (sentence_tokens) and bigrams (sentence_bigrams)
    :param sentence: string
    :return: list, list
        sentence_tokens: ordered list of words found in the sentence
        sentence_bigrams: a list of ordered two-word tuples found in the sentence
    """
    sentence_tokens = ['<s>'] + sentence.lower().split() + ['</s>']
    sentence_bigrams = []
    for i in range(len(sentence_tokens)-1):
        sentence_bigrams.append((sentence_tokens[i], sentence_tokens[i+1]))
    return sentence_tokens, sentence_bigrams

def find_bigram_count(sentences):
    bg_dict = defaultdict(dict)
    for sentence in sentences:
        split_sentence = sentence.split()
        for i in range(len(split_sentence)-1):
            first_word = split_sentence[i]
            second_word = split_sentence[i+1]
            if first_word in bg_dict.keys():
                if second_word in bg_dict[first_word].keys():
                    bg_dict[first_word][second_word]+=1
                else:
                    bg_dict[first_word][second_word] = 1
            else:
                bg_dict[first_word][second_word] = 1
    return bg_dict

def bigram_suggesstions(first_word, input_dataset):
    bigram = find_bigram_count(input_dataset)
    tokens, _ = bigrams_from_sentences(input_dataset)
    token_counts = Counter(tokens)
    token_val = token_counts[first_word]
    suggestions = {}
    for b in bigram[first_word]:
        suggestions[b] = bigram[first_word][b]/float(token_val)
    return suggestions
def bigram_add1_logs(transcript_file):
    """
    provide a smoothed log probability dictionary based on a transcript
    :param transcript_file: string
        transcript_file is the path filename containing unpunctuated text sentences
    :return: dict
        bg_add1_log_dict: dictionary of smoothed bigrams log probabilities including
        tags: <s>: start of sentence, </s>: end of sentence, <unk>: unknown placeholder probability
    """

    #tokens, bigrams = bigrams_from_transcript(transcript_file)
    tokens, bigrams = bigrams_from_sentences(transcript_file)
    token_counts = Counter(tokens)
    bigram_counts = Counter(bigrams)
    vocab_count = len(token_counts)

    bg_addone_dict = {}
    for bg in bigram_counts:
        bg_addone_dict[bg] = np.log((bigram_counts[bg] + 1.) / (token_counts[bg[0]] + vocab_count))
    bg_addone_dict['<unk>'] = np.log(1. / vocab_count)
    return bg_addone_dict


test_sentences = [
    'the old man spoke to me',
    'me to spoke man old the',
    'old man me old man me',
]

def sample_run(sentences):
    # sample usage by test code (this definition not actually run for the quiz)
    #bigram_log_dict = utils.bigram_add1_logs('transcripts.txt')
    bigram_log_dict = bigram_add1_logs(generate_corpus(desc_file='train_clean_corpus.json'))
    for sentence in sentences:
        print('*** "{}"'.format(sentence))
        print(log_prob_of_sentence(sentence, bigram_log_dict))

def log_prob_of_sentence(sentence, bigram_log_dict):
    total_log_prob = 0.

    # TODO implement
    # get the sentence bigrams with utils.sentence_to_bigrams
    tokens, bigrams = sentence_to_bigrams(sentence)
    for bigram in bigrams:
        if bigram in bigram_log_dict:
            total_log_prob = total_log_prob + bigram_log_dict[bigram]
        else:
            total_log_prob = total_log_prob + bigram_log_dict["<unk>"]
    return total_log_prob


def sample_bigram_dict_run(desc_file):
    # sample usage by test code (this definition not actually run for the quiz)
    tokens, bigrams = bigrams_from_sentences(generate_corpus(desc_file=desc_file))
    bg_dict = bigram_mle(tokens, bigrams)
    #print(bg_dict)
    return bg_dict


def bigram_mle(tokens, bigrams):
    """
    provide a dictionary of probabilities for all bigrams in a corpus of text
    the calculation is based on maximum likelihood estimation and does not include
    any smoothing.  A tag '<unk>' has been added for unknown probabilities.
    :param tokens: list
        tokens: list of all tokens in the corpus
    :param bigrams: list
        bigrams: list of all two word tuples in the corpus
    :return: dict
        bg_mle_dict: a dictionary of bigrams:
            key: tuple of two bigram words, in order OR <unk> key
            value: float probability

    """
    bg_mle_dict = {}
    bg_mle_dict['<unk>'] = 0.
    #TODO implement
    c_bigrams = Counter(bigrams)
    c_tokens = Counter(tokens)
    for key, val in c_bigrams.items():
        if(c_tokens[key[0]] > 0):
            bg_mle_dict[key] = val/c_tokens[key[0]]
    #print(bg_mle_dict)
    #print(c_tokens.keys())
    return bg_mle_dict