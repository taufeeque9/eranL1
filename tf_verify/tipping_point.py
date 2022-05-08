import os
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image', type=int, default=-1, help='Image idx')
args = parser.parse_args()

def evaluate(eps, dataset):
    os.system('rm -rf results.txt')
    ret = os.system(f' python3 . --netname l1model_combined_v2.onnx --domain deeppoly --complete True --epsilon {eps} --dataset {dataset} --image {args.image}  >> /dev/null 2>&1')
    if ret:
        exit()
    with open('results.txt', 'r') as rf:
        return int(rf.read().strip())

def get_tipping_point(a, b, dataset, ea=None, eb=None):
    print(a, b)
    if b-a < 0.001:
        return b
    if ea is None:
        ea = evaluate(a, dataset)
    if eb is None:
        eb = evaluate(b, dataset)
    if ea != eb:
        em = evaluate((a+b)/2, dataset)
        if em == ea:
            return get_tipping_point((a+b)/2, b, dataset, em, eb)
        else:
            return get_tipping_point(a, (a+b)/2, dataset, ea, em)
    else:
        print(f'ea, eb, same error - a = {a} & b = {b} & e = {ea}')
        exit()

t = get_tipping_point(0.01, 0.08, 'mnist')
print('  mnist -', t)
t = get_tipping_point(0.01, 0.08, 'l1mnist')
print('l1mnist -', t)
