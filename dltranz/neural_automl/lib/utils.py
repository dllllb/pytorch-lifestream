import os
import glob
import hashlib
import gc
import time
import numpy as np
import contextlib
import torch
import tqdm


def iterate_minibatches(*tensors, batch_size, shuffle=True, epochs=1,
                        allow_incomplete=True, callback=lambda x:x):
    indices = np.arange(len(tensors[0]))
    upper_bound = int((np.ceil if allow_incomplete else np.floor) (len(indices) / batch_size)) * batch_size
    epoch = 0
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in callback(range(0, upper_bound, batch_size)):
            batch_ix = indices[batch_start: batch_start + batch_size]
            batch = [tensor[batch_ix] for tensor in tensors]
            yield batch if len(tensors) > 1 else batch[0]
        epoch += 1
        if epoch >= epochs:
            break


def process_in_chunks(function, *args, batch_size, out=None, **kwargs):
    """
    Computes output by applying batch-parallel function to large data tensor in chunks
    :param function: a function(*[x[indices, ...] for x in args]) -> out[indices, ...]
    :param args: one or many tensors, each [num_instances, ...]
    :param batch_size: maximum chunk size processed in one go
    :param out: memory buffer for out, defaults to torch.zeros of appropriate size and type
    :returns: function(data), computed in a memory-efficient way
    """
    total_size = args[0].shape[0]
    first_output = function(*[x[0: batch_size] for x in args])
    output_shape = (total_size,) + tuple(first_output.shape[1:])
    if out is None:
        out = torch.zeros(*output_shape, dtype=first_output.dtype, device=first_output.device,
                          layout=first_output.layout, **kwargs)

    out[0: batch_size] = first_output
    with tqdm.tqdm(total=int(total_size/batch_size)) as steps:
        for i in range(batch_size, total_size, batch_size):
            batch_ix = slice(i, min(i + batch_size, total_size))
            out[batch_ix] = function(*[x[batch_ix] for x in args])

            steps.set_description(f"    test processing... ")
            steps.update()
        return out


def check_numpy(x):
    """ Makes sure x is a numpy array """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


@contextlib.contextmanager
def nop_ctx():
    yield None


def get_latest_file(pattern):
    list_of_files = glob.glob(pattern) # * means all if need specific format then *.csv
    assert len(list_of_files) > 0, "No files found: " + pattern
    return max(list_of_files, key=os.path.getctime)


def md5sum(fname):
    """ Computes mdp checksum of a file """
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def free_memory(sleep_time=0.1):
    """ Black magic function to free torch memory and some jupyter whims """
    gc.collect()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(sleep_time)


def to_float_str(element):
    try:
        return str(float(element))
    except ValueError:
        return element
