import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader

from tools import get_device, logging_base_config
from tools.dataset import ImageDataset, load_dataset
from tools.loss import MSELossBalanced
from tools.params import load_model
from tools.shape_generator import ShapeGenerator
from tools.train import valid

num_processes = 1
params = {
    'valid_path': "../datasets/supervised-valid-4000.p",
    'num_workers': 16,
    'valid_batch_size': 64,

    'model_path_load_from': '../models/ml-model-color-48-bs128-tuning-80.p',
}


def show_model_result(n_obj, n_sample, model):
    fig, axs = plt.subplots(n_obj, n_sample * 2 + 2, figsize=(2 * (n_sample * 2 + 2), 2 * n_obj))

    model.eval()

    for row in range(n_obj):
        shape = ShapeGenerator(keep_params=list(ShapeGenerator.params_pattern.keys()))
        for col in range(n_sample):
            ax = axs[row, col]
            ax.axis('off')
            ax.imshow(shape.get_buffer()[0])

        X, y = shape.get_buffer()
        ax = axs[row, n_sample]
        ax.axis('off')
        ax.text(0, 0, '\n'.join(['{}: [{}]'.format(k, ', '.join([f'{i:.2f}' for i in v]))
                                 for k, v in sorted(y.items())]))

        x = torch.from_numpy(X.reshape((1,) + X.shape))
        x = x.to(device)
        with torch.no_grad():
            pred_params = {k: v.cpu().numpy()[0] for k, v in model(x).items()}
        pred_shape = ShapeGenerator(keep_params=[], params=pred_params)

        for col in range(n_sample):
            ax = axs[row, n_sample + 1 + col]
            ax.axis('off')
            ax.imshow(pred_shape.get_buffer()[0])

        ax = axs[row, n_sample * 2 + 1]
        ax.axis('off')
        ax.text(0, 0, '\n'.join(['{}: [{}]'.format(k, ', '.join([f'{i:.2f}' for i in v]))
                                 for k, v in sorted(pred_params.items())]))

    plt.show()


if __name__ == '__main__':
    logging_base_config()
    device = get_device()

    pretrained_model = load_model(**params)

    valid_data = ImageDataset(load_dataset(params['valid_path']))
    valid_loader = DataLoader(valid_data, batch_size=params['valid_batch_size'],
                              shuffle=True, num_workers=params['num_workers'])
    loss_fn = MSELossBalanced()
    score = valid(
        seed=42,
        test_loader=valid_loader,
        device=device,
        model=pretrained_model,
        loss_fn=loss_fn,
    )
    print(f'RESULTS: {score}')

    # Evaluate
    show_model_result(12, 3, pretrained_model)
