import torch
from torch.nn import Linear, BatchNorm1d, ReLU
import numpy as np
from dltranz.tabnet import sparsemax

"""
Based on pytorch tabnet implementation: https://github.com/dreamquark-ai/tabnet
"""


def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim+output_dim)/np.sqrt(4*input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim+output_dim)/np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


class GBN(torch.nn.Module):
    """
        Ghost Batch Normalization
        https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)


class TabNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim,
                 n_d=8, n_a=8,
                 n_steps=3, gamma=1.3,
                 n_independent=2, n_shared=2, epsilon=1e-15,
                 virtual_batch_size=128, momentum=0.02,
                 mask_type="sparsemax"):
        """
        Defines TabNet network

        Parameters
        ----------
        input_dim : int
            Number of features
        output_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of sucessive steps in the newtork (usually betwenn 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually betwenn 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.initial_bn = BatchNorm1d(self.input_dim, momentum=0.01)

        if self.n_shared > 0:
            shared_feat_transform = torch.nn.ModuleList()
            for i in range(self.n_shared):
                if i == 0:
                    shared_feat_transform.append(Linear(self.input_dim,
                                                        2*(n_d + n_a),
                                                        bias=False))
                else:
                    shared_feat_transform.append(Linear(n_d + n_a, 2*(n_d + n_a), bias=False))

        else:
            shared_feat_transform = None

        self.initial_splitter = FeatTransformer(self.input_dim, n_d+n_a, shared_feat_transform,
                                                n_glu_independent=self.n_independent,
                                                virtual_batch_size=self.virtual_batch_size,
                                                momentum=momentum)

        self.feat_transformers = torch.nn.ModuleList()
        self.att_transformers = torch.nn.ModuleList()

        for step in range(n_steps):
            transformer = FeatTransformer(self.input_dim, n_d+n_a, shared_feat_transform,
                                          n_glu_independent=self.n_independent,
                                          virtual_batch_size=self.virtual_batch_size,
                                          momentum=momentum)
            attention = AttentiveTransformer(n_a, self.input_dim,
                                             virtual_batch_size=self.virtual_batch_size,
                                             momentum=momentum,
                                             mask_type=self.mask_type)
            self.feat_transformers.append(transformer)
            self.att_transformers.append(attention)

        if self.is_multi_task:
            self.multi_task_mappings = torch.nn.ModuleList()
            for task_dim in output_dim:
                task_mapping = Linear(n_d, task_dim, bias=False)
                initialize_non_glu(task_mapping, n_d, task_dim)
                self.multi_task_mappings.append(task_mapping)
        else:
            self.final_mapping = Linear(n_d, output_dim, bias=False)
            initialize_non_glu(self.final_mapping, n_d, output_dim)

    def forward(self, x):
        res = 0
        x = self.initial_bn(x)

        prior = torch.ones(x.shape).to(x.device)
        # M_loss = 0
        att = self.initial_splitter(x)[:, self.n_d:]

        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            # M_loss += torch.mean(torch.sum(torch.mul(M, torch.log(M+self.epsilon)), dim=1))
            # update prior
            prior = torch.mul(self.gamma - M, prior)
            # output
            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            d = ReLU()(out[:, :self.n_d])
            res = torch.add(res, d)
            # update attention
            att = out[:, self.n_d:]

        # M_loss /= self.n_steps

        if self.is_multi_task:
            # Result will be in list format
            out = []
            for task_mapping in self.multi_task_mappings:
                out.append(task_mapping(res))
        else:
            out = self.final_mapping(res)
        return out  # , M_loss

    def forward_masks(self, x):
        x = self.initial_bn(x)

        prior = torch.ones(x.shape).to(x.device)
        M_explain = torch.zeros(x.shape).to(x.device)
        att = self.initial_splitter(x)[:, self.n_d:]
        masks = {}

        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            masks[step] = M
            # update prior
            prior = torch.mul(self.gamma - M, prior)
            # output
            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            d = ReLU()(out[:, :self.n_d])
            # explain
            step_importance = torch.sum(d, dim=1)
            M_explain += torch.mul(M, step_importance.unsqueeze(dim=1))
            # update attention
            att = out[:, self.n_d:]

        return M_explain, masks


class AttentiveTransformer(torch.nn.Module):
    def __init__(self, input_dim, output_dim,
                 virtual_batch_size=128,
                 momentum=0.02,
                 mask_type="sparsemax"):
        """
        Initialize an attention transformer.

        Parameters
        ----------
        input_dim : int
            Input size
        output_dim : int
            Outpu_size
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(AttentiveTransformer, self).__init__()
        self.fc = Linear(input_dim, output_dim, bias=False)
        initialize_non_glu(self.fc, input_dim, output_dim)
        self.bn = GBN(output_dim, virtual_batch_size=virtual_batch_size,
                      momentum=momentum)

        if mask_type == "sparsemax":
            # Sparsemax
            self.selector = sparsemax.Sparsemax(dim=-1)
        elif mask_type == "entmax":
            # Entmax
            self.selector = sparsemax.Entmax15(dim=-1)
        else:
            raise NotImplementedError("Please choose either sparsemax" +
                                      "or entmax as masktype")

    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.mul(x, priors)
        x = self.selector(x)
        return x


class FeatTransformer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, shared_layers, n_glu_independent,
                 virtual_batch_size=128, momentum=0.02):
        super(FeatTransformer, self).__init__()
        """
        Initialize a feature transformer.

        Parameters
        ----------
        input_dim : int
            Input size
        output_dim : int
            Outpu_size
        shared_layers : torch.nn.ModuleList
            The shared block that should be common to every step
        n_glu_independant : int
            Number of independent GLU layers
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization within GLU block(s)
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        """

        params = {
            'n_glu': n_glu_independent,
            'virtual_batch_size': virtual_batch_size,
            'momentum': momentum
        }

        if shared_layers is None:
            # no shared layers
            self.shared = torch.nn.Identity()
            is_first = True
        else:
            self.shared = GLU_Block(input_dim, output_dim,
                                    first=True,
                                    shared_layers=shared_layers,
                                    n_glu=len(shared_layers),
                                    virtual_batch_size=virtual_batch_size,
                                    momentum=momentum)
            is_first = False

        if n_glu_independent == 0:
            # no independent layers
            self.specifics = torch.nn.Identity()
        else:
            spec_input_dim = input_dim if is_first else output_dim
            self.specifics = GLU_Block(spec_input_dim, output_dim,
                                       first=is_first,
                                       **params)

    def forward(self, x):
        x = self.shared(x)
        x = self.specifics(x)
        return x


class GLU_Block(torch.nn.Module):
    """
        Independant GLU block, specific to each step
    """

    def __init__(self, input_dim, output_dim, n_glu=2, first=False, shared_layers=None,
                 virtual_batch_size=128, momentum=0.02):
        super(GLU_Block, self).__init__()
        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = torch.nn.ModuleList()

        params = {
            'virtual_batch_size': virtual_batch_size,
            'momentum': momentum
        }

        fc = shared_layers[0] if shared_layers else None
        self.glu_layers.append(GLU_Layer(input_dim, output_dim,
                                         fc=fc,
                                         **params))
        for glu_id in range(1, self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(GLU_Layer(output_dim, output_dim,
                                             fc=fc,
                                             **params))

    def forward(self, x):
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))
        if self.first:  # the first layer of the block has no scale multiplication
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x*scale
        return x


class GLU_Layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, fc=None,
                 virtual_batch_size=128, momentum=0.02):
        super(GLU_Layer, self).__init__()

        self.output_dim = output_dim
        if fc:
            self.fc = fc
        else:
            self.fc = Linear(input_dim, 2*output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2*output_dim)

        self.bn = GBN(2*output_dim, virtual_batch_size=virtual_batch_size,
                      momentum=momentum)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, :self.output_dim], torch.sigmoid(x[:, self.output_dim:]))
        return out
