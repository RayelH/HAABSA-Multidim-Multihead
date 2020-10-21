import os
import json
import xml.etree.ElementTree as ET
from collections import Counter
import string
import en_core_web_sm

en_nlp = en_core_web_sm.load()
import nltk
import re
import numpy as np


def window(iterable, size):  # stack overflow solution for sliding window
    i = iter(iterable)
    win = []
    for e in range(0, size):
        win.append(next(i))
    yield win
    for e in i:
        win = win[1:] + [e]
        yield win


def _get_data_tuple(sptoks, asp_termIn, label):
    # Find the ids of aspect term
    aspect_is = []
    asp_term = ' '.join(sp for sp in asp_termIn).lower()
    for _i, group in enumerate(window(sptoks, len(asp_termIn))):
        if asp_term == ' '.join([g.lower() for g in group]):
            aspect_is = list(range(_i, _i + len(asp_termIn)))
            break
        elif asp_term in ' '.join([g.lower() for g in group]):
            aspect_is = list(range(_i, _i + len(asp_termIn)))
            break

    print(aspect_is)
    pos_info = []
    for _i, sptok in enumerate(sptoks):
        pos_info.append(min([abs(_i - i) for i in aspect_is]))

    lab = None
    if label == 'negative':
        lab = -1
    elif label == 'neutral':
        lab = 0
    elif label == "positive":
        lab = 1
    else:
        raise ValueError("Unknown label: %s" % lab)

    return pos_info, lab


"""
This function reads data from the xml file

Iput arguments:
@fname: file location
@source_count: list that contains list [<pad>, 0] at the first position [empty input]
and all the unique words with number of occurences as tuples [empty input]
@source_word2idx: dictionary with unique words and unique index [empty input]
.. same for target

Return:
@source_data: list with lists which contain the sentences corresponding to the aspects saved by word indices 
@target_data: list which contains the indices of the target phrases: THIS DOES NOT CORRESPOND TO THE INDICES OF source_data 
@source_loc_data: list with lists which contains the distance from the aspect for every word in the sentence corresponding to the aspect
@target_label: contains the polarity of the aspect (0=negative, 1=neutral, 2=positive)
@max_sen_len: maximum sentence length
@max_target_len: maximum target length

"""


def read_data_2016(fname, source_count, source_word2idx, target_count, target_phrase2idx, file_name):
    if os.path.isfile(fname) == False:
        raise ("[!] Data %s not found" % fname)

    # parse xml file to tree
    tree = ET.parse(fname)
    root = tree.getroot()

    outF = open(file_name, "w")

    # save all words in source_words (includes duplicates)
    # save all aspects in target_words (includes duplicates)
    # finds max sentence length and max targets length
    source_words, target_words, max_sent_len, max_target_len = [], [], 0, 0
    all_aspect_categories = []
    all_unique_aspect_categories_count = []
    all_polarities = []
    all_polarities_count = []
    target_phrases = []

    number_opinions = 0
    all_number_opinions = []
    opinions_count = []


    count_explicit_opinions = 0
    count_all_opinions = 0


    countConfl = 0
    for sentence in root.iter('sentence'):
        number_opinions = 0
        sent = sentence.find('text').text
        # if there is more than 1 space, replace it by single space
        sentenceNew = re.sub(' +', ' ', sent)
        sptoks = nltk.word_tokenize(sentenceNew) # tokenized sentences
        for sp in sptoks:
            source_words.extend([''.join(sp).lower()]) # put every word from every sentence in source words
        if len(sptoks) > max_sent_len:
            max_sent_len = len(sptoks) # get length of longest sentence
        for opinions in sentence.iter('Opinions'): # get opinions tag from sentence (<Opinions>)
            for opinion in opinions.findall('Opinion'): # loop over all opinions (<opinion>)
                count_all_opinions += 1
                if opinion.get("polarity") == "conflict": # skip if polarity is conflict
                    countConfl += 1
                    continue
                asp = opinion.get('target') # get aspect
                aspect_category = opinion.get('category')
                polarity = opinion.get('polarity')
                if asp != 'NULL': # skip implicit aspects
                    number_opinions += 1
                    count_explicit_opinions += 1
                    aspNew = re.sub(' +', ' ', asp)
                    all_aspect_categories.extend([''.join(aspect_category)])
                    all_polarities.extend([''.join(polarity)])
                    t_sptoks = nltk.word_tokenize(aspNew)
                    for sp in t_sptoks:
                        target_words.extend([''.join(sp).lower()]) # put all target words in target_words
                    target_phrases.append(' '.join(sp for sp in t_sptoks).lower()) # add al target phrases to the list
                    if len(t_sptoks) > max_target_len:
                        max_target_len = len(t_sptoks) # get max target length
        all_number_opinions.extend([''.join(str(number_opinions))])

    if len(source_count) == 0:
        source_count.append(['<pad>', 0])
    source_count.extend(Counter(source_words + target_words).most_common())
    target_count.extend(Counter(target_phrases).most_common())
    all_unique_aspect_categories_count.extend(Counter(all_aspect_categories).most_common())
    all_polarities_count.extend(Counter(all_polarities).most_common())
    opinions_count.extend(Counter(all_number_opinions).most_common())
    # print(target_count)
    # print(source_count)
    # print(target_count)
    # print(all_unique_aspect_categories_count)
    # print(count_all_opinions)
    # print(count_explicit_opinions)
    # print(all_polarities_count)
    # print(opinions_count)

    return count_all_opinions, count_explicit_opinions


def main():
    source_count, target_count = [], []
    source_word2idx, target_phrase2idx = {}, {}
    total_opinions_train, explicit_opinions_train = read_data_2016()
    total_opinions_test, explicit_opinions_test = read_data_2016()
    # target_train_set = set(target_train)
    # target_test_set = set(target_test)
    #
    # print(target_train_set)
    # print(len(target_train_set))
    # print(target_test_set)
    # print(target_test_set - target_train_set)
    # print(len(target_test_set - target_train_set))
    print(total_opinions_train)
    print(total_opinions_test)
    print(explicit_opinions_train)
    print(explicit_opinions_test)
if __name__ == '__main__':
    main()
