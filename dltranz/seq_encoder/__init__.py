from dltranz.seq_encoder.agg_feature_model import AggFeatureSeqEncoder
from dltranz.seq_encoder.rnn_encoder import RnnSeqEncoder, RnnSeqEncoderDistributionTarget
from dltranz.seq_encoder.transf_seq_encoder import TransfSeqEncoder
from dltranz.seq_encoder.statistics_encoder import StatisticsEncoder
from dltranz.seq_encoder.dummy_encoder import DummyEncoder


def create_encoder(params, is_reduce_sequence):
    encoder_type = params['encoder_type']
    if encoder_type == 'rnn':
        return RnnSeqEncoder(params, is_reduce_sequence)
    if encoder_type == 'transf':
        return TransfSeqEncoder(params, is_reduce_sequence)
    if encoder_type == 'agg_features':
        return AggFeatureSeqEncoder(params, is_reduce_sequence)
    if encoder_type == 'statistics':
        return StatisticsEncoder(params)
    if encoder_type == 'distribution_targets':
        return RnnSeqEncoderDistributionTarget(params, is_reduce_sequence)
    if encoder_type == 'emb_valid':
        return DummyEncoder(params)

    raise AttributeError(f'Unknown encoder_type: "{encoder_type}"')
