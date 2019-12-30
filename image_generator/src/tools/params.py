import json

import torch


def load_model(model_path_load_from, **_):
    if model_path_load_from is None:
        return None
    model = torch.load(model_path_load_from)
    print(f'Model loaded from "{model_path_load_from}"')
    return model


def save_model(model, result, model_path_save_to, **params):
    torch.save(model, model_path_save_to)
    print(f'Model saved to "{model_path_save_to}"')

    with open(model_path_save_to + '.json', 'w') as f:
        json.dump({
            'result': result,
            'params': params,
        }, f, indent=2)
    print(f'Results:\n{json.dumps(result, indent=2)}')
