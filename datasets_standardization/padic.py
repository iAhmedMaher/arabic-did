#!/usr/bin/env python
# coding: utf-8
import xml.etree.ElementTree as ET
import os
from lang_trans.arabic import buckwalter
import pandas as pd
import numpy as np

# SUG: add main and argparse

padic_dir = r'..\..\datasets\PADIC'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir(os.path.join(padic_dir, 'eval'))
ensure_dir(os.path.join(padic_dir, 'train'))
ensure_dir(os.path.join(padic_dir, 'test'))

tree = ET.parse(os.path.join(padic_dir, 'PADIC.xml'))

root = tree.getroot()

padic_list = []
node_label_pairs = [('MOROCCAN', 'MOR'), ('ANNABA', 'ANN'), ('MODERN-STANDARD-ARABIC', 'MSA'),
                    ('SYRIAN', 'SYR'), ('PALESTINIAN', 'PAL'), ('ALGIERS', 'ALG')]
for sentence in root:
    for node, label in node_label_pairs:
        padic_list += [label, buckwalter.untransliterate(sentence.find(node).text[3:])]

df = pd.DataFrame(np.array(padic_list).reshape(-1,2), columns=['label', 'text'])
df.index.name = 'id'

# SUG: random sampling is better
train = df.iloc[:25968, :]
evaluate = df.iloc[25968:25968+8652, :]
test = df.iloc[25968+8652:25968+8652+8652, :]

train.to_csv(os.path.join(padic_dir, 'train', 'train.csv'))
evaluate.to_csv(os.path.join(padic_dir, 'eval', 'eval.csv'))
test.to_csv(os.path.join(padic_dir, 'test', 'test.csv'))
