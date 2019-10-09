#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
from torch.nn.functional import one_hot
import torch

# TODO: normalize arabic presentation forms

username_re = re.compile(r'(?<=^|(?<=[^a-zA-Z0-9-_]))@([A-Za-z]+[A-Za-z0-9_]+)')
url_re = re.compile(r'((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
hashtag_re = re.compile(r'#(\w+)')
diacritics_re = re.compile(r'[\u064b-\u065f\u06d4-\u06ed\u08d5-\u08ff]')
non_character_set_re = re.compile(r'[^\u0620-\u064a.!?,٬٫ ]+')
numbers_re = re.compile(r'[0-9\u0660-\u0669]+')

class TextPreprocessor(object):
    def __init__(self, config):
        self.labels_to_int = config['labels_to_int']
        self.max_seq_len = config['preprocessing']['max_seq_len']
        self.tokenization = config['preprocessing']['tokenization']
        self.normalize = config['preprocessing']['normalize']
        self.max_rep = config['preprocessing']['max_rep']
        self.token2int_dict = self.get_char2int_dict() if self.tokenization == 'char' \
            else self.get_word2int_dict()
        print('token2int dict: ', self.token2int_dict)

    def get_num_tokens(self):
        return len(self.token2int_dict)

    def get_char2int_dict(self):
        all_chars = ' '.join([chr(c) for c in range(2**16)])
        all_possible_chars = self.prepare_text(all_chars)
        char2int = {'pad' : 0}  # ALERT: Padding assumes the zeroth place is pad
        count = 1
        for c in all_possible_chars:
            if c not in char2int:
                char2int[c] = count
                count += 1
        return char2int

    # TODO: implement word tokenization
    def get_word2int_dict(self):
        pass

    def clean_text(self, text):
        text = username_re.sub('الشخص', text)
        text = url_re.sub('الرابط', text)
        text = hashtag_re.sub('', text)
        text = numbers_re.sub('ألف', text)
        text = diacritics_re.sub('', text)
        text = non_character_set_re.sub(' ', text)
        text = re.sub('[ ]+', ' ', text)
        return text

    def normalize_arabic(self, text):
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("[ؿؾؠؽ]", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("گ", "ك", text)
        return text

    def set_max_rep(self, text, rep=3):
        new_text = text[0]
        i = 1
        while i < len(text):
            if text[i] == text[i - 1]:
                c = 1
                while text[i] == text[i - 1]:
                    i += 1
                    c += 1
                new_text += min(rep - 1, c) * text[i - 1]
            else:
                new_text += text[i]
                i += 1
        return new_text

    # TODO allow removing punctuation
    def prepare_text(self, text):
        text = self.clean_text(text)
        if self.normalize:
            text = self.normalize_arabic(text)
        if self.max_rep != 0:
            text = self.set_max_rep(text, self.max_rep)

        if self.tokenization == 'char':
            return self.char_tokenize(text)
        else:
            self.word_tokenize(text)
            return text

    def char_tokenize(self, text):
        return [c for c in text]

    def word_tokenize(self, text):
        return text.split()

    def standardize_tokens_length(self, int_tokenized_text):
        if len(int_tokenized_text) > self.max_seq_len:
            return int_tokenized_text[:self.max_seq_len]

        return int_tokenized_text + [0] * (self.max_seq_len - len(int_tokenized_text))

    def __call__(self, batch):
        int_labels = [self.labels_to_int[pair[0]] for pair in batch]
        int_labels_tensor = torch.LongTensor(int_labels)
        one_hot_labels = one_hot(int_labels_tensor, len(self.labels_to_int))

        tokenized_texts = [self.prepare_text(pair[1]) for pair in batch]
        int_tokenized_texts = [ [self.token2int_dict[c] for c in tokenized_text] for tokenized_text in tokenized_texts]
        int_tokenized_texts_tensor = torch.LongTensor([self.standardize_tokens_length(int_toks)
                                                       for int_toks in int_tokenized_texts])
        int_tokenized_texts_tensor.permute(1,0) # Pytorch prefers sequence batches to be T, B, F

        return one_hot_labels, int_tokenized_texts_tensor

