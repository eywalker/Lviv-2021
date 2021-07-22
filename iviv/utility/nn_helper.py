import random
import numpy as np
import torch

def set_random_seed(seed: int, deterministic: bool = True):
    """
    Set random generator seed for Python interpreter, NumPy and PyTorch. When setting the seed for PyTorch,
    if CUDA device is available, manual seed for CUDA will also be set. Finally, if `deterministic=True`,
    and CUDA device is available, PyTorch CUDNN backend will be configured to `benchmark=False` and `deterministic=True`
    to yield as deterministic result as possible. For more details, refer to
    PyTorch documentation on reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    Beware that the seed setting is a "best effort" towards deterministic run. However, as detailed in the above documentation,
    there are certain PyTorch CUDA opertaions that are inherently non-deterministic, and there is no simple way to control for them.
    Thus, it is best to assume that when CUDA is utilized, operation of the PyTorch module will not be deterministic and thus
    not completely reproducible.
    Args:
        seed (int): seed value to be set
        deterministic (bool, optional): If True, CUDNN backend (if available) is set to be deterministic. Defaults to True. Note that if set
            to False, the CUDNN properties remain untouched and it NOT explicitly set to False.
    """
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)  # this sets both CPU and CUDA seeds for PyTorch

def get_io_dims(data_loader):
    """
    Returns the shape of the dataset for each item within an entry returned by the `data_loader`
    The DataLoader object must return either a namedtuple, dictionary or a plain tuple.
    If `data_loader` entry is a namedtuple or a dictionary, a dictionary with the same keys as the
    namedtuple/dict item is returned, where values are the shape of the entry. Otherwise, a tuple of
    shape information is returned.
    Note that the first dimension is always the batch dim with size depending on the data_loader configuration.
    Args:
        data_loader (torch.DataLoader): is expected to be a pytorch Dataloader object returning
            either a namedtuple, dictionary, or a plain tuple.
    Returns:
        dict or tuple: If data_loader element is either namedtuple or dictionary, a ditionary
            of shape information, keyed for each entry of dataset is returned. Otherwise, a tuple
            of shape information is returned. The first dimension is always the batch dim
            with size depending on the data_loader configuration.
    """
    items = next(iter(data_loader))
    if hasattr(items, "_asdict"):  # if it's a named tuple
        items = items._asdict()

    if hasattr(items, "items"):  # if dict like
        return {k: v.shape for k, v in items.items()}
    else:
        return (v.shape for v in items)
        
def get_dims_for_loader_dict(dataloaders):
    """
    Given a dictionary of DataLoaders, returns a dictionary with same keys as the
    input and shape information (as returned by `get_io_dims`) on each keyed DataLoader.
    Args:
        dataloaders (dict of DataLoader): Dictionary of dataloaders.
    Returns:
        dict: A dict containing the result of calling `get_io_dims` for each entry of the input dict
    """
    return {k: get_io_dims(v) for k, v in dataloaders.items()}
