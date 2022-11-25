import random
import numpy as np

def sequence_pair_augmentation(item, max_lenght=np.inf):
    length = len(next(iter(item.values())))
    l_length = random.randint(length // 4, min(3 * length // 4, max_lenght // 2))
    left = {k: v[:l_length] for k, v in item.items()}
    right = {k: v[l_length:] for k, v in item.items()}
    return left, right  
