import random


def sequence_pair_augmentation(item):
    length = len(next(iter(item.values())))
    l_length = random.randint(length // 4, 3 * length // 4)
    left = {k: v[:l_length] for k, v in item.items()}
    right = {k: v[l_length:] for k, v in item.items()}

    return left, right
