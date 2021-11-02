import torch.nn

from dltranz.lightning_modules.stream_encoder import StreamEncoder


def test_validation_step_predict():
    se = StreamEncoder(
        encoder_x2z=torch.nn.Linear(4, 2),
        history_size=5, predict_range=[0], predict_w=[1],
        z_channels=2, c_channels=2,
        var_gamma_z=0.5, var_gamma_c=0.3,
        lr=0.001, weight_decay=0.0, step_size=1, gamma=0.9,
        cpc_w=1.0, cpc_neg_w=1.0, cov_z_w=1.0, var_z_w=1.0, cov_c_w=1.0, var_c_w=1.0,
    )

    batch = torch.randn(5, 10, 4),
    se.validation_step(batch, 0)


def test_validation_step_predict_range():
    se = StreamEncoder(
        encoder_x2z=torch.nn.Linear(4, 2),
        history_size=5, predict_range=[0, 2, -2], predict_w=[1/3] * 3,
        z_channels=2, c_channels=2,
        var_gamma_z=0.5, var_gamma_c=0.3,
        lr=0.001, weight_decay=0.0, step_size=1, gamma=0.9,
        cpc_w=1.0, cpc_neg_w=1.0, cov_z_w=1.0, var_z_w=1.0, cov_c_w=1.0, var_c_w=1.0,
    )

    batch = torch.randn(3, 20, 4),
    se.validation_step(batch, 0)


def test_cpc_loss():
    se = StreamEncoder(
        encoder_x2z=torch.nn.Linear(4, 2),
        history_size=5, predict_range=[0], predict_w=[1],
        z_channels=2, c_channels=2,
        lr=0.001, weight_decay=0.0, step_size=1, gamma=0.9,
        cpc_w=1.0, cpc_neg_w=1.0, cov_z_w=1.0, var_z_w=1.0, cov_c_w=1.0, var_c_w=1.0,
    )

    x = torch.randn(5, 5, 2)
    y = torch.randn(5, 1, 2)
    loss, neg_loss, cx = se.cpc_loss(x, torch.cat([y, x], dim=1))
    assert cx.shape == (5, 2)


class TestDummyRnn(torch.nn.Module):
    def forward(self, x):
        return x, None


def test_cpc_neg_loss():
    se = StreamEncoder(
        encoder_x2z=torch.nn.Linear(4, 2),
        history_size=4, predict_range=[0], predict_w=[1],
        z_channels=2, c_channels=2,
        lr=0.001, weight_decay=0.0, step_size=1, gamma=0.9,
        cpc_w=1.0, cpc_neg_w=1.0, cov_z_w=1.0, var_z_w=1.0, cov_c_w=1.0, var_c_w=1.0,
    )

    se.ar_rnn_z2c = TestDummyRnn()
    se.lin_predictors_c2p[0].weight.data = torch.eye(2)
    se.lin_predictors_c2p[0].bias.data.zero_()

    x1 = torch.tensor([
        [
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ],
        [
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
        ]
    ]).float()
    y1 = torch.tensor([
        [
            [0, 1],
        ],
        [
            [1, 0],
        ]
    ]).float()
    x2 = torch.tensor([
        [
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ],
        [
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ]
    ]).float()
    y2 = torch.tensor([
        [
            [0, 1],
        ],
        [
            [0, 1],
        ]
    ]).float()

    loss1, neg_loss1, cx1 = se.cpc_loss(x1, torch.cat([y1, x1], dim=1))
    loss2, neg_loss2, cx2 = se.cpc_loss(x2, torch.cat([y2, x2], dim=1))

    assert neg_loss1 < neg_loss2
