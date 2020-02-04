import pytest
import torch

from dltranz.seq_encoder import scoring_head, PaddedBatch, PerTransHead, TimeStepShuffle, SkipStepEncoder, RnnEncoder
from dltranz.transf_seq_encoder import TransformerSeqEncoder


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
