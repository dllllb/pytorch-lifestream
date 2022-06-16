import numpy as np

from ptls.nn.seq_encoder import RnnSeqEncoder


class RnnSeqEncoderDistributionTarget(RnnSeqEncoder):
    def transform(self, x):
        return np.sign(x) * np.log(np.abs(x) + 1)

    def transform_inv(self, x):
        return np.sign(x) * (np.exp(np.abs(x)) - 1)

    def __init__(self,
                 trx_encoder,
                 hidden_size,
                 type,
                 bidir,
                 trainable_starter,
                 head_layers,
                 input_size=None,
                 ):
        super().__init__(
            trx_encoder,
            input_size,
            hidden_size,
            type,
            bidir,
            trainable_starter,
        )
        head_params = dict(head_layers).get('CombinedTargetHeadFromRnn', None)
        self.pass_samples = head_params.get('pass_samples', True)
        self.numeric_name = list(trx_encoder.scalers.keys())[0]
        self.collect_pos, self.collect_neg = (head_params.get('pos', True), head_params.get('neg', True)) if head_params else (0, 0)
        self.eps = 1e-7

    def forward(self, x):
        amount_col = []
        for i, row in enumerate(x.payload[self.numeric_name]):
            amount_col += [list(row[:x.seq_lens[i].item()].cpu().numpy())]
        amount_col = np.array(amount_col, dtype=object)
        neg_sums = []
        pos_sums = []
        for list_row in amount_col:
            np_row = np.array(list_row)
            if self.collect_neg:
                neg_sums += [np.sum(self.transform_inv(np_row[np.where(np_row < 0)]))]
            if self.collect_pos:
                pos_sums += [np.sum(self.transform_inv(np_row[np.where(np_row >= 0)]))]
        neg_sum_logs = np.log(np.abs(np.array(neg_sums)) + self.eps)
        pos_sum_logs = np.log(np.array(pos_sums) + self.eps)

        x = super().forward(x)
        if (not self.pass_samples):
            return x
        if self.collect_neg and self.collect_pos:
            return x, neg_sum_logs, pos_sum_logs
        elif self.collect_neg:
            return x, neg_sum_logs
        elif self.collect_pos:
            return x, pos_sum_logs