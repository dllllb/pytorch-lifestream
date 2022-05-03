import pytest
import torch

from ptls.seq_encoder.rnn_encoder import RnnEncoder, SkipStepEncoder
from ptls.seq_encoder.utils import PerTransHead, TimeStepShuffle, scoring_head
from ptls.seq_encoder.transf_seq_encoder import TransformerSeqEncoder
from ptls.trx_encoder import PaddedBatch


class TrxEncoderTest(torch.nn.Module):
    def forward(self, x):
        return x


def tst_rnn_model(config):
    p = TrxEncoderTest()
    e = RnnEncoder(8, config['rnn'])
    h = scoring_head(16, config['head'])
    m = torch.nn.Sequential(p, e, h)
    return m


def tst_trx_avg_model():
    p = TrxEncoderTest()
    h = PerTransHead(8)
    m = torch.nn.Sequential(p, h)
    return m


def tst_transf_model(config):
    p = TrxEncoderTest()
    e = TransformerSeqEncoder(8, config['transf'])
    h = scoring_head(8, config['head'])
    m = torch.nn.Sequential(p, e, h)
    return m


def test_simple_config():
    config = {
        'rnn': {
            'trainable_starter': 'empty',
            'hidden_size': 16,
            'type': 'gru',
            'bidir': False,
        },
        'head': {
            'explicit_lengths': False,
            'pred_all_states': False,
            'pred_all_states_mean': False,
            'norm_input': False,
            'use_batch_norm': False,
        }
    }

    m = tst_rnn_model(config)

    x = torch.rand(12, 100, 8)
    length = torch.arange(0, 12, 1)
    x = PaddedBatch(x, length)
    out = m(x)

    assert len(out.size()) == 1
    assert out.size()[0] == 12


def test_concat_lens():
    config = {
        'rnn': {
            'trainable_starter': 'empty',
            'hidden_size': 16,
            'type': 'gru',
            'bidir': False,
        },
        'head': {
            'explicit_lengths': True,
            'pred_all_states': False,
            'pred_all_states_mean': False,
            'norm_input': False,
            'use_batch_norm': False,
        },
    }
    rnn = tst_rnn_model(config)

    x = torch.rand(12, 100, 8)
    length = torch.arange(0, 12, 1)
    x = PaddedBatch(x, length)
    out = rnn(x)

    assert len(out.size()) == 1
    assert out.size()[0] == 12


def test_trainable_starter():
    config = {
        'rnn': {
            'trainable_starter': 'static',
            'hidden_size': 16,
            'type': 'gru',
            'bidir': False,
        },
        'head': {
            'explicit_lengths': True,
            'pred_all_states': False,
            'pred_all_states_mean': False,
            'norm_input': False,
            'use_batch_norm': False,
        },
    }
    rnn = tst_rnn_model(config)

    x = torch.rand(12, 100, 8)
    length = torch.arange(0, 12, 1)
    x = PaddedBatch(x, length)
    out = rnn(x)

    assert len(out.size()) == 1
    assert out.size()[0] == 12


def test_rnn_type():
    config = {
        'rnn': {
            'trainable_starter': 'empty',
            'hidden_size': 16,
            'type': 'gru',
            'bidir': False,
        },
        'head': {
            'explicit_lengths': False,
            'pred_all_states': False,
            'pred_all_states_mean': False,
            'norm_input': False,
            'use_batch_norm': False,
        },
    }
    rnn = tst_rnn_model(config)

    x = torch.rand(12, 100, 8)
    length = torch.arange(0, 12, 1)
    x = PaddedBatch(x, length)
    out = rnn(x)

    assert len(out.size()) == 1
    assert out.size()[0] == 12


def test_pred_all_states():
    config = {
        'rnn': {
            'trainable_starter': 'empty',
            'hidden_size': 16,
            'type': 'gru',
            'bidir': False,
        },
        'head': {
            'explicit_lengths': False,
            'pred_all_states': True,
            'pred_all_states_mean': False,
            'norm_input': False,
            'use_batch_norm': False,
        },
    }

    rnn = tst_rnn_model(config)

    x = torch.rand(12, 100, 8)
    length = torch.arange(0, 12, 1)
    x = PaddedBatch(x, length)
    out = rnn(x).payload

    assert len(out.size()) == 2
    assert out.size()[0] == 12


def test_pred_all_states_mean():
    config = {
        'rnn': {
            'trainable_starter': 'empty',
            'hidden_size': 16,
            'type': 'gru',
            'bidir': False,
        },
        'head': {
            'explicit_lengths': False,
            'pred_all_states': True,
            'pred_all_states_mean': True,
            'norm_input': False,
            'use_batch_norm': False,
        },
    }
    rnn = tst_rnn_model(config)

    x = torch.rand(12, 100, 8)
    length = torch.arange(0, 12, 1)
    x = PaddedBatch(x, length)
    out = rnn(x)

    assert len(out.size()) == 1
    assert out.size()[0] == 12


def test_pred_all_states_no_effect():
    x = torch.rand(12, 100, 8)
    length = torch.arange(0, 12, 1)
    x = PaddedBatch(x, length)

    config = {
        'rnn': {
            'trainable_starter': 'empty',
            'hidden_size': 16,
            'type': 'gru',
            'bidir': False,
        },
        'head': {
            'explicit_lengths': False,
            'pred_all_states': False,
            'pred_all_states_mean': False,
            'norm_input': False,
            'use_batch_norm': False,
        },
    }
    rnn = tst_rnn_model(config)
    out1 = rnn(x)

    rnn.pred_all_states = True
    out2 = rnn(x)

    assert (out1 == out2).all()


def test_pred_all_states_and_concat_lens():
    config = {
        'rnn': {
            'trainable_starter': 'empty',
            'hidden_size': 16,
            'type': 'gru',
            'bidir': False,
        },
        'head': {
            'explicit_lengths': True,
            'pred_all_states': True,
            'pred_all_states_mean': False,
            'norm_input': False,
            'use_batch_norm': False,
        },
    }

    with pytest.raises(AttributeError):
        tst_rnn_model(config)


def test_all():
    x = torch.rand(12, 100, 8)
    length = torch.arange(0, 12, 1) + 1
    x = PaddedBatch(x, length)

    for concat_lens in [False, True]:
        for trainable_starter in ['empty', 'static']:
            for rnn_type in ['gru', 'lstm']:
                for pred_all_states in [False, True]:
                    for pred_all_states_mean in [False, True]:
                        #
                        if pred_all_states and concat_lens:
                            continue
                        if pred_all_states and not pred_all_states_mean:
                            expected_size = 2
                        else:
                            expected_size = 1

                        config = {
                            'rnn': {
                                'trainable_starter': trainable_starter,
                                'hidden_size': 16,
                                'type': rnn_type,
                                'bidir': False,
                            },
                            'head': {
                                'explicit_lengths': concat_lens,
                                'pred_all_states': pred_all_states,
                                'pred_all_states_mean': pred_all_states_mean,
                                'norm_input': False,
                                'use_batch_norm': False,
                            },
                        }
                        rnn = tst_rnn_model(config)
                        try:
                            out = rnn(x)

                            if isinstance(out, PaddedBatch):
                                out = out.payload

                            assert len(out.size()) == expected_size
                            assert out.size()[0] == 12
                        except Exception:
                            print(config)
                            raise


def test_trx_avg_encoder():
    m = tst_trx_avg_model()

    x = torch.rand(12, 100, 8)
    length = torch.arange(0, 12, 1)
    x = PaddedBatch(x, length)
    out = m(x)

    assert len(out.size()) == 1
    assert out.size()[0] == 12


def test_timestep_shuffle():
    t = torch.tensor([
        [[0, 0], [1, 2], [3, 4], [0, 0]],
        [[0, 0], [10, 11], [0, 0], [0, 0]],
    ])

    res = TimeStepShuffle()(PaddedBatch(t, [2, 1]))

    assert res.payload.shape == (2, 4, 2)


def test_skip_step_encoder():
    t = torch.arange(8*11*2).view(8, 11, 2)

    res = SkipStepEncoder(3)(PaddedBatch(t, [10, 9, 8, 7, 3, 2, 1, 0]))

    assert res.payload.shape == (8, 4, 2)


def test_transf_seq_encoder():
    config = {
        'transf': {
            'n_heads': 2,
            'dim_hidden': 16,
            'dropout': .1,
            'n_layers': 2,
            'max_seq_len': 200,
            'use_after_mask': True,
            'use_positional_encoding': True,
            'sum_output': True,
            'input_size': 32,
            'shared_layers': False,
            'use_src_key_padding_mask': False,
            'train_starter': False
        },
        'head': {
            'explicit_lengths': False,
            'pred_all_states': False,
            'pred_all_states_mean': False,
            'norm_input': False,
            'use_batch_norm': False,
        },
    }

    m = tst_transf_model(config)

    x = torch.rand(12, 100, 8)
    length = torch.arange(0, 12, 1)
    x = PaddedBatch(x, length)
    out = m(x)

    assert len(out.size()) == 1
    assert out.size()[0] == 12


def test_rnn_iterative_no_starter():
    """
    x_a - clients with 2 parts of transactions: x_a_1 and x_a_2
    x_b_2 - clients with only one second part or transactions

    We run inference in 2 steps:
    x_a_1          -> out_1
    x_a_1 + x_b_2  -> out_2

    Also we run an other kind os splits, for `a` and `b` groups
    x_a   -> out_a
    x_b_2 -> out_b

    We expect the same results in specific positions of out_1, out_2 and out_a, out_b

    :return:
    """
    GRP_A_CLIENT_COUNT = 3
    GRP_B_CLIENT_COUNT = 6
    PART_1_TRX_COUNT = 7
    PART_2_TRX_COUNT = 2
    INPUT_SIZE = 4
    OUTPUT_SIZE = 5

    conf = {
        'hidden_size': OUTPUT_SIZE,
        'type': 'gru',
        'bidir': False,
        'trainable_starter': 'none',
    }

    m = RnnEncoder(INPUT_SIZE, conf)
    m.eval()
    print(m)

    x_a_1 = torch.rand(GRP_A_CLIENT_COUNT, PART_1_TRX_COUNT, INPUT_SIZE)
    x_a_2 = torch.rand(GRP_A_CLIENT_COUNT, PART_2_TRX_COUNT, INPUT_SIZE)
    x_b_2 = torch.rand(GRP_B_CLIENT_COUNT, PART_2_TRX_COUNT, INPUT_SIZE)

    out_a = m(PaddedBatch(torch.cat([x_a_1, x_a_2], dim=1), None))
    out_b = m(PaddedBatch(x_b_2, None))
    out_1 = m(PaddedBatch(x_a_1, None))
    starter = torch.cat([out_1.payload[:, -1, :], torch.zeros(GRP_B_CLIENT_COUNT, OUTPUT_SIZE)], dim=0)
    out_2 = m(PaddedBatch(torch.cat([x_a_2, x_b_2], dim=0), None), starter.unsqueeze(0))

    out_a_merged = torch.cat([out_1.payload, out_2.payload[:GRP_A_CLIENT_COUNT]], dim=1)
    assert ((out_a.payload - out_a_merged).abs() < 1e-4).all()

    out_b_merged = out_2.payload[GRP_A_CLIENT_COUNT:]
    assert ((out_b.payload - out_b_merged).abs() < 1e-4).all()


def test_rnn_iterative_with_starter():
    """
    x_a - clients with 2 parts of transactions: x_a_1 and x_a_2
    x_b_2 - clients with only one second part or transactions

    We run inference in 2 steps:
    x_a_1          -> out_1
    x_a_1 + x_b_2  -> out_2

    Also we run an other kind os splits, for `a` and `b` groups
    x_a   -> out_a
    x_b_2 -> out_b

    We expect the same results in specific positions of out_1, out_2 and out_a, out_b

    :return:
    """
    GRP_A_CLIENT_COUNT = 3
    GRP_B_CLIENT_COUNT = 6
    PART_1_TRX_COUNT = 7
    PART_2_TRX_COUNT = 2
    INPUT_SIZE = 4
    OUTPUT_SIZE = 5

    conf = {
        'hidden_size': OUTPUT_SIZE,
        'type': 'gru',
        'bidir': False,
        'trainable_starter': 'static',
    }

    m = RnnEncoder(INPUT_SIZE, conf)
    m.eval()
    print(m)

    x_a_1 = torch.rand(GRP_A_CLIENT_COUNT, PART_1_TRX_COUNT, INPUT_SIZE)
    x_a_2 = torch.rand(GRP_A_CLIENT_COUNT, PART_2_TRX_COUNT, INPUT_SIZE)
    x_b_2 = torch.rand(GRP_B_CLIENT_COUNT, PART_2_TRX_COUNT, INPUT_SIZE)

    out_a = m(PaddedBatch(torch.cat([x_a_1, x_a_2], dim=1), None))
    out_b = m(PaddedBatch(x_b_2, None))
    out_1 = m(PaddedBatch(x_a_1, None))
    starter = torch.cat([out_1.payload[:, -1, :], torch.zeros(GRP_B_CLIENT_COUNT, OUTPUT_SIZE)], dim=0)
    out_2 = m(PaddedBatch(torch.cat([x_a_2, x_b_2], dim=0), None), starter.unsqueeze(0))

    out_a_merged = torch.cat([out_1.payload, out_2.payload[:GRP_A_CLIENT_COUNT]], dim=1)
    assert ((out_a.payload - out_a_merged).abs() < 1e-4).all()

    out_b_merged = out_2.payload[GRP_A_CLIENT_COUNT:]
    assert ((out_b.payload - out_b_merged).abs() < 1e-4).all()
