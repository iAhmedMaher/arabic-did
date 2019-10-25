import os
import pandas as pd
import numpy as np

shami_dir = r'..\..\datasets\Shami'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir(os.path.join(shami_dir, 'eval'))
ensure_dir(os.path.join(shami_dir, 'train'))
ensure_dir(os.path.join(shami_dir, 'test'))

file_label_tuple = [(r'..\datasets\Shami\syrian.txt', 'SYR'),
                   (r'..\datasets\Shami\Palestinian.txt', 'PAL'),
                   (r'..\datasets\Shami\Lebanees.txt', 'LEB'),
                   (r'..\datasets\Shami\jordinian.txt', 'JOR')]

shami_list = []
for file, label in file_label_tuple:
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            shami_list += [label, line.rstrip()]

df = pd.DataFrame(np.array(shami_list).reshape(-1,2), columns=['label', 'text'])
df.index.name = 'id'
df = df.sample(frac=1).reset_index(drop=True)
df.index.name = 'id'
train_size = len(df)*0.7
eval_size = len(df)*0.15
test_size = len(df)*0.15

current = 0

train = df.loc[current:train_size+current, :]
current += len(train)

evaluate = df.loc[current:eval_size+current, :]
current += len(evaluate)

test = df.loc[current:test_size+current, :]
current += len(test)

train.to_csv(os.path.join(shami_dir, 'train', 'train.csv'))
evaluate.to_csv(os.path.join(shami_dir, 'eval', 'eval.csv'))
test.to_csv(os.path.join(shami_dir, 'test', 'test.csv'))
