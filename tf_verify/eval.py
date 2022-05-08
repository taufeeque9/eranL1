import os
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epsilon', type=float, default=0.055, help='the epsilon for L_infinity perturbation')
parser.add_argument('--n', type=int, default=5, help='number of images')
args = parser.parse_args()

os.system('rm -rf results.txt')

for i in tqdm(range(args.n)):
    ret = os.system(f'python3 . --netname l1model_combined.onnx --domain deeppoly --complete True --epsilon {args.epsilon} --dataset mnist --image {i}  >> /dev/null 2>&1')
    # print(ret)
    if ret:
        exit()

with open('results.txt', 'r') as rf:
    results_mnist = rf.readlines()
    results_mnist = np.array([int(e.strip()) for e in results_mnist])

os.system('mv results.txt results_mnist.txt')

for i in tqdm(range(args.n)):
    ret = os.system(f' python3 . --netname l1model_combined.onnx --domain deeppoly --complete True --epsilon {args.epsilon} --dataset l1mnist --image {i}  >> /dev/null 2>&1')
    if ret:
        exit()

with open('results.txt', 'r') as rf:
    results = rf.readlines()
    results = np.array([int(e.strip()) for e in results])
    print('  mnist:', 100*results_mnist.mean())
    print('l1mnist:', 100*results.mean())

os.system('mv results.txt results_l1mnist.txt')
