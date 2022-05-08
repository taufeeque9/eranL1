import os
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

eps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
mnist = [96, 84, 50, 14, 4, 2, 2]
l1mnist = [96, 90, 70, 48, 18, 16, 10]

plt.plot(eps, mnist, label='L_inf')
plt.plot(eps, l1mnist, label='L_1')
plt.xlabel('Epsilon')
plt.ylabel('Verification Accuracy')
plt.tight_layout()
plt.legend()
plt.savefig('eps_linf_vs_l1.png')
