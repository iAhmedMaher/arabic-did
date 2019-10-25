#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
from torch.nn.functional import one_hot
import torch
from helpers import get_datasets_paths
import pandas as pd
import numpy as np
from collections import Counter

# TODO: normalize arabic presentation forms

username_re = re.compile(r'(?<=^|(?<=[^a-zA-Z0-9-_]))@([A-Za-z]+[A-Za-z0-9_]+)')
url_re = re.compile(r'((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
hashtag_re = re.compile(r'#(\w+)')
diacritics_re = re.compile(r'[\u064b-\u065f\u06d4-\u06ed\u08d5-\u08ff]')
non_character_set_re = re.compile(r'[^\u0620-\u064a.!?,٬٫ ]+')
numbers_re = re.compile(r'[0-9\u0660-\u0669]+')

class TextPreprocessor(object):
    def __init__(self, config, return_text=False):
        self.labels_to_int = config['labels_to_int']
        self.tokenization = config['preprocessing']['tokenization']
        self.normalize = config['preprocessing']['normalize']
        self.max_rep = config['preprocessing']['max_rep']
        self.token2int_dict = self.get_char2int_dict() if self.tokenization == 'char' \
            else self.get_word2int_dict(config)
        self.return_text = return_text

    def get_num_tokens(self):
        return len(self.token2int_dict)

    def get_char2int_dict(self):
        all_chars = ' '.join([chr(c) for c in range(2**16)])
        all_possible_chars = self.tokenize_text(self.process_text(all_chars))
        char2int = {'pad' : 0}  # ALERT: Padding assumes the zeroth place is pad
        count = 1
        for c in all_possible_chars:
            if c not in char2int:
                char2int[c] = count
                count += 1
        return char2int

    # TODO allow different word tokenization techniques
    def get_word2int_dict(self, config):
        csv_paths_list = get_datasets_paths(config, 'train')
        combined_dataset = None
        for csv_path in csv_paths_list:
            current_dataset = pd.read_csv(csv_path, index_col='id')
            combined_dataset = pd.concat([combined_dataset, current_dataset], axis=0)

        token_dict = {}
        for label in self.labels_to_int.keys():
            text_lists = list(combined_dataset[combined_dataset.label == label]['text'])
            all_words = []
            for row in text_lists:
                all_words += self.tokenize_text(self.process_text(row))
            counts = Counter(all_words)
            token_dict.update(dict(counts.most_common(int(config['preprocessing']['per_class_vocab_size']))))

        for idx, token in enumerate(token_dict):
            token_dict[token] = idx + 1 # +1 because pad will be 0

        token_dict['pad'] = 0
        token_dict['oov'] = len(token_dict)

        return token_dict

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
    def process_text(self, text):
        text = self.clean_text(text)
        if self.normalize:
            text = self.normalize_arabic(text)
        if self.max_rep != 0:
            text = self.set_max_rep(text, self.max_rep)

        return text

    def tokenize_text(self, text):
        if self.tokenization == 'char':
            return self.char_tokenize(text)
        elif self.tokenization == 'word':
            self.word_tokenize(text)
            return text
        else:
            raise NotImplementedError()

    def char_tokenize(self, text):
        return [c for c in text]

    def word_tokenize(self, text):
        return text.split(' ')

    def standardize_tokens_length(self, int_tokenized_text, max_seq_len):
        if len(int_tokenized_text) > max_seq_len:
            return int_tokenized_text[:max_seq_len]

        return int_tokenized_text + [0] * (max_seq_len - len(int_tokenized_text))

    def token2int(self, token):
        try:
            return self.token2int_dict[token]
        except KeyError:
            return self.token2int_dict['oov']

    def __call__(self, batch):
        int_labels = [self.labels_to_int[pair[0]] for pair in batch]
        int_labels_tensor = torch.LongTensor(int_labels)

        original_texts = [pair[1] for pair in batch]
        processed_texts = [self.process_text(text) for text  in original_texts]
        tokenized_texts = [self.tokenize_text(text) for text in processed_texts]
        int_tokenized_texts = [ [self.token2int(c) for c in tokenized_text] for tokenized_text in tokenized_texts]
        max_seq_len = max([len(int_tokenized_text) for int_tokenized_text in int_tokenized_texts])
        int_tokenized_texts_tensor = torch.LongTensor([self.standardize_tokens_length(int_toks, max_seq_len)
                                                       for int_toks in int_tokenized_texts])
        int_tokenized_texts_tensor = int_tokenized_texts_tensor.permute(1,0) # Pytorch prefers sequence batches to be T, B, F

        if self.return_text:
            return int_labels_tensor, int_tokenized_texts_tensor, original_texts, processed_texts
        else:
            return int_labels_tensor, int_tokenized_texts_tensor