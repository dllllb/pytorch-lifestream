import torch

def gen_trx_data(lengths, target_share=.5):
    n = len(lengths)
    targets = (torch.rand(n) >= target_share).long()
    samples = list()
    for target, length in zip(targets, lengths):
        s = dict()
        s['trans_type'] = (torch.rand(length)*10 + 1).long()
        s['mcc_code'] = (torch.rand(length) * 20 + 1).long()
        s['amount'] = (torch.rand(length) * 1000 + 1).long()

        samples.append({'feature_arrays': s, 'target': target})

    return samples
