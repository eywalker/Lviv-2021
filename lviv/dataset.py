from collections import OrderedDict
from itertools import zip_longest

import numpy as np
from neuralpredictors.data.datasets import FileTreeDataset
from neuralpredictors.data.samplers import SubsetSequentialSampler
from neuralpredictors.data.transforms import NeuroNormalizer, Subsample, ToTensor

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .utility.data_helper import get_oracle_dataloader
from .utility.nn_helper import set_random_seed


def load_dataset(
    path,
    batch_size,
    areas=None,
    layers=None,
    tier=None,
    neuron_ids=None,
    neuron_n=None,
    exclude_neuron_n=0,
    neuron_base_seed=None,
    image_ids=None,
    image_n=None,
    image_base_seed=None,
    get_key=False,
    cuda=True,
    normalize=True,
    exclude="images",
    return_test_sampler=False,
    oracle_condition=None,
    seed=None,
):
    """
    returns a single data loader
    Args:
        path (str): path for the dataset
        batch_size (int): batch size.
        areas (list, optional): the visual area.
        layers (list, optional): the layer from visual area.
        tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader.
        neuron_ids (list, optional): select neurons by their ids.
        neuron_n (int, optional): number of neurons to select randomly. Can not be set together with neuron_ids
        neuron_base_seed (float, optional): base seed for neuron selection. Get's multiplied by neuron_n to obtain final seed
        exclude_neuron_n (int): the first <exclude_neuron_n> neurons will be excluded (given a neuron_base_seed),
                                then <neuron_n> neurons will be drawn from the remaining neurons.
        image_ids (list, optional): select images by their ids.
        image_n (int, optional): number of images to select randomly. Can not be set together with image_ids
        image_base_seed (float, optional): base seed for image selection. Get's multiplied by image_n to obtain final seed
        get_key (bool, optional): whether to return the data key, along with the dataloaders.
        cuda (bool, optional): whether to place the data on gpu or not.
        normalize (bool, optional): whether to normalize the data (see also exclude)
        exclude (str, optional): data to exclude from data-normalization. Only relevant if normalize=True. Defaults to 'images'
        include_behavior (bool, optional): whether to include behavioral data
        file_tree (bool, optional): whether to use the file tree dataset format. If False, equivalent to the HDF5 format
        return_test_sampler (bool, optional): whether to return only the test loader with repeat-batches
        oracle_condition (list, optional): Only relevant if return_test_sampler=True. Class indices for the sampler
    Returns:
        if get_key is False returns a dictionary of dataloaders for one dataset, where the keys are 'train', 'validation', and 'test'.
        if get_key is True it returns the data_key (as the first output) followed by the dataloder dictionary.
    """
    if seed is not None:
        set_random_seed(seed)
    assert any(
        [image_ids is None, all([image_n is None, image_base_seed is None])]
    ), "image_ids can not be set at the same time with any other image selection criteria"
    assert any(
        [
            neuron_ids is None,
            all(
                [
                    neuron_n is None,
                    neuron_base_seed is None,
                    areas is None,
                    layers is None,
                    exclude_neuron_n == 0,
                ]
            ),
        ]
    ), "neuron_ids can not be set at the same time with any other neuron selection criteria"
    assert any(
        [exclude_neuron_n == 0, neuron_base_seed is not None]
    ), "neuron_base_seed must be set when exclude_neuron_n is not 0"

    data_key = (
        path.split("static")[-1]
        .split(".")[0]
        .replace("preproc", "")
        .replace("_nobehavior", "")
    )

    dat = FileTreeDataset(path, "images", "responses")

    # The permutation MUST be added first and the conditions below MUST NOT be based on the original order
    # specify condition(s) for sampling neurons. If you want to sample specific neurons define conditions that would effect idx
    conds = np.ones(len(dat.neurons.area), dtype=bool)
    if areas is not None:
        conds &= np.isin(dat.neurons.area, areas)
    if layers is not None:
        conds &= np.isin(dat.neurons.layer, layers)
    idx = np.where(conds)[0]
    if neuron_n is not None:
        random_state = np.random.get_state()
        if neuron_base_seed is not None:
            np.random.seed(
                neuron_base_seed * neuron_n
            )  # avoid nesting by making seed dependent on number of neurons
        assert (
            len(dat.neurons.unit_ids) >= exclude_neuron_n + neuron_n
        ), "After excluding {} neurons, there are not {} neurons left".format(
            exclude_neuron_n, neuron_n
        )
        neuron_ids = np.random.choice(
            dat.neurons.unit_ids, size=exclude_neuron_n + neuron_n, replace=False
        )[exclude_neuron_n:]
        np.random.set_state(random_state)
    if neuron_ids is not None:
        idx = [
            np.where(dat.neurons.unit_ids == unit_id)[0][0] for unit_id in neuron_ids
        ]

    more_transforms = [Subsample(idx), ToTensor(cuda)]
    if normalize:
        more_transforms.insert(0, NeuroNormalizer(dat, exclude=exclude))
    dat.transforms.extend(more_transforms)

    if return_test_sampler:
        print("Returning only test sampler with repeats...")
        dataloader = get_oracle_dataloader(dat, oracle_condition=oracle_condition)
        return (data_key, {"test": dataloader}) if get_key else {"test": dataloader}

    # subsample images
    dataloaders = {}
    keys = [tier] if tier else ["train", "validation", "test"]
    tier_array = dat.trial_info.tiers
    image_id_array = dat.trial_info.frame_image_id
    for tier in keys:
        # sample images
        if tier == "train" and image_ids is not None:
            subset_idx = [
                np.where(image_id_array == image_id)[0][0] for image_id in image_ids
            ]
            assert (
                sum(tier_array[subset_idx] != "train") == 0
            ), "image_ids contain validation or test images"
        elif tier == "train" and image_n is not None:
            random_state = np.random.get_state()
            if image_base_seed is not None:
                np.random.seed(
                    image_base_seed * image_n
                )  # avoid nesting by making seed dependent on number of images
            subset_idx = np.random.choice(
                np.where(tier_array == "train")[0], size=image_n, replace=False
            )
            np.random.set_state(random_state)
        else:
            subset_idx = np.where(tier_array == tier)[0]

        sampler = (
            SubsetRandomSampler(subset_idx)
            if tier == "train"
            else SubsetSequentialSampler(subset_idx)
        )
        dataloaders[tier] = DataLoader(dat, sampler=sampler, batch_size=batch_size)

    # create the data_key for a specific data path
    return (data_key, dataloaders) if get_key else dataloaders
