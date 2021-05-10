from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
import os

warnings.filterwarnings('ignore', category=FutureWarning)
import torch
import random
import numpy as np

# Create outputs and checkpoint folders for first run
if not os.path.exists('experiments/ckpt'):
    os.makedirs('experiments/ckpt')
if not os.path.exists('experiments/outputs'):
    os.makedirs('experiments/outputs')


# Fixes for reproducible results
def fix_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fix_seed()
