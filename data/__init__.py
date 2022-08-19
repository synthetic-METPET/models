"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


def mix_create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = MixCustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data


class MixCustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)    # MUST: First half is cropped, back half is whole.
        print("dataset [%s] was created" % type(self.dataset).__name__)

        self.total_num_cropped = self.dataset.total_num_cropped
        self.total_num_whole = self.dataset.total_num_whole


        if (self.opt.phase == 'train') or (self.opt.phase == 'val'):
            from torch.utils.data.sampler import WeightedRandomSampler
            assert (self.opt.mix_ratio is not None), 'No mix ratio.'
            self.ratio_cropped = int((self.opt.mix_ratio).split(':')[0])   # eg. '8:2' -> 8  croppped
            self.ratio_whole = int((self.opt.mix_ratio).split(':')[1])     # eg. '8:2' -> 2  whole
            assert ((self.ratio_cropped > 0) and (self.ratio_whole > 0)), 'Cannot receive zero ratio.'
            #assert (self.ratio_cropped + self.ratio_whole == 10), 'The total of mix ratio exceeds 10.'
    
            # Make sampler
            _samples_weight = ([self.ratio_cropped] * self.total_num_cropped) + ([self.ratio_whole] * self.total_num_whole)
            sampler = WeightedRandomSampler(_samples_weight, len(_samples_weight)) 
            self.dataloader = torch.utils.data.DataLoader(
                                                        self.dataset,
                                                        batch_size=opt.batch_size,
                                                        shuffle = False, # not opt.serial_batches,
                                                        num_workers=int(opt.num_threads),
                                                        sampler=sampler)
        else:
            # if (self.opt.phase == 'test')
            self.dataloader = torch.utils.data.DataLoader(
                                                        self.dataset,
                                                        batch_size=opt.batch_size,
                                                        shuffle = False, # not opt.serial_batches,
                                                        num_workers=int(opt.num_threads)
                                                        )  # no sampler

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        max_dataset_size = self.total_num_cropped  # == self.total_whole  if necessary, self.total_cropped + self.total_whole
        num_cropped_in_epoch = 0
        num_whole_in_epoch = 0
        for i, data in enumerate(self.dataloader):
            A_paths = data['A_paths']   # == AB_paths
            num_cropped_in_batch = sum('cropped' in path.split('/') for path in A_paths) #  ..../train/whole/00001466_T1-Axi_CE+_FS+_~_PET_Brain_slice_pair_3ch_1ch_0025_cropped.png
            num_whole_in_batch = sum('whole' in path.split('/') for path in A_paths)     #  ..../train/cropped/00001466_T1-Axi_CE+_FS+_~_PET_Brain_slice_pair_3ch_1ch_0025_whole.png
            num_cropped_in_epoch = num_cropped_in_epoch + num_cropped_in_batch
            num_whole_in_epoch = num_whole_in_epoch + num_whole_in_batch
            total_num_in_epoch = num_cropped_in_epoch + num_whole_in_epoch

            #if total_num_in_epoch >= max_dataset_size:
            #    break
            yield data

            if (self.opt.phase == 'train') or (self.opt.phase == 'val'):
                # Until exceeds, continue to yiled.
                if total_num_in_epoch >= max_dataset_size:
                    break
            else:
                pass
