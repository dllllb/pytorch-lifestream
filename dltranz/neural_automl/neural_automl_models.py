import dltranz.neural_automl.lib as lib
import torch
import torch .nn as nn
import os
import json


def get_model(model_config, data=None, input_size=None):
    if model_config is None or not isinstance(model_config, dict) or not model_config.get('layers', False):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf/default_config.json'), 'r') as f:
            model_config = json.load(f)

    return get_model_from_config(model_config, data, input_size)


def get_model_from_config(model_config, data=None, input_size=None):

    def check(l, key):
        if not l.get(key, False):
            raise KeyError(f'not found required parameter {key}')

    def get(l, key):
        check(l, key)
        return l[key]

    def get_paramsl(l, lst_keys):
        return [get(l, key) for key in lst_keys]

    def get_prev_layer_output():
        layer, aggregations = None, []
        for i in range(len(lst_layers_config)):
            l = lst_layers_config[-(i + 1)]

            if l['type'] == 'sigmoid' or l['type'] == 'relu':
                continue

            elif l['type'] == 'aggregation':
                dims = l.get('dims', -1)
                aggregations.append(dims)

            elif l['type'] == 'tree' or l['type'] == 'linear':
                layer = l
                break

            else:
                raise NotImplemented(f"unsupported layer type automatik size infer {l['type']}")

        if layer is None:
            raise LookupError('no previous layer found for automatik size infere')

        out_shape = None
        if layer['type'] == 'tree':
            num_trees, num_sub_layers, tree_depth, tree_dim = get_paramsl(layer,
                                                                          ['num_trees', 'num_sub_layers',
                                                                           'tree_depth', 'tree_dim'])
            flatten_output = layer.get('flatten_output', 'False')
            if flatten_output:
                out_shape = [num_trees * num_sub_layers * tree_dim]
            else:
                out_shape = [num_trees * num_sub_layers, tree_dim]

        elif layer['type'] == 'linear':
            out_shape = [layer['d_out']]

        for aggr in aggregations:
            if isinstance(aggr, int):
                aggr = [aggr]
            aggr = [k if k >= 0 else len(out_shape) + k for k in aggr]

            out_shape = [out_shape[idx] for idx in range(len(out_shape)) if idx not in aggr]

        if len(out_shape) == 1:
            return out_shape[0]
        elif len(out_shape) == 0:
            return 1 # constant
        else:
            return out_shape

    lst_layers, lst_layers_config = [], []
    for i, layer in enumerate(model_config['layers']):
        check(layer, 'type')

        if layer['type'] == 'tree':
            num_trees, num_sub_layers, tree_depth, tree_dim = get_paramsl(layer, ['num_trees', 'num_sub_layers', 'tree_depth', 'tree_dim'])
            # get d_in
            d_in = layer.get('d_in', False)
            if i == 0 and not d_in:
                if data is not None or input_size is not None:
                    d_in = input_size if input_size is not None else data.shape[-1]
                else:
                    check(d_in)
            if i > 0 and not d_in:
                d_in = get_prev_layer_output()

            flatten_output = layer.get('flatten_output', 'True')

            lst_layers.append(lib.DenseBlock(d_in,
                                             num_trees,
                                             num_layers=num_sub_layers,
                                             tree_dim=tree_dim,
                                             depth=tree_depth,
                                             flatten_output=flatten_output,
                                             choice_function=lib.entmax15,
                                             bin_function=lib.entmoid15))

        elif layer['type'] == 'linear':
            d_out = get(layer, 'd_out')
            # get d_in
            d_in = layer.get('d_in', False)
            if i == 0 and not d_in:
                if data is not None or input_size is not None:
                    d_in = input_size if input_size is not None else data.shape[-1]
                else:
                    check(d_in)
            if i > 0 and not d_in:
                d_in = get_prev_layer_output()

            bias = True if 'bias' not in layer else layer['bias']
            lst_layers.append(nn.Linear(d_in, d_out, bias=bias))

        elif layer['type'] == 'aggregation':
            func = get(layer, 'func')
            dims = layer.get('dims', -1)

            if func == 'mean':
                lst_layers.append(lib.Lambda(lambda x: x.mean(dim=dims)))
            elif func == 'sum':
                lst_layers.append(lib.Lambda(lambda x: x.sum(dim=dims)))
            else:
                raise NotImplemented(f"unknown aggregation function type {func}")

        elif layer['type'] == 'sigmoid' or layer['type'] == 'relu':
            if layer['type'] == 'sigmoid':
                lst_layers.append(nn.Sigmoid())
            elif layer['type'] == 'relu':
                lst_layers.append(nn.ReLU())

        else:
            raise not NotImplemented(f"unknown layer type: {layer['type']}")

        # remember already added layers config
        lst_layers_config.append(layer)

    lst_layers.append(lib.Lambda(lambda x: x.squeeze()))

    device = torch.device(model_config.get('device', 'cpu'))
    model = nn.Sequential(*lst_layers).to(device)

    if data is not None:
        n_samples = model_config.get('data_aware_initialisation', 10000)
        with torch.no_grad():
            _ = model(torch.as_tensor(data[:n_samples], device=device).float())

    return model
