"""
Module for Pytorch dataset representations
"""

import torch
from torch.utils.data import Dataset

class SlicesDataset(Dataset):
    """
    This class represents an indexable Torch dataset
    which could be consumed by the PyTorch DataLoader class
    """
    def __init__(self, data):
        self.data = data

        self.slices = []

        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                self.slices.append((i, j))

    def __getitem__(self, idx):
        """
        This method is called by PyTorch DataLoader class to return a sample with id idx

        Arguments: 
            idx {int} -- id of sample

        Returns:
            Dictionary of 2 Torch Tensors of dimensions [1, W, H]
        """
        slc = self.slices[idx]
        sample = dict()
        sample["id"] = idx

        # You could implement caching strategy here if dataset is too large to fit
        # in memory entirely
        # Also this would be the place to call transforms if data augmentation is used

        # TASK: Create two new keys in the "sample" dictionary, named "image" and "seg"
        # The values are 3D Torch Tensors with image and label data respectively. 
        # First dimension is size 1, and last two hold the voxel data from the respective
        # slices. Write code that stores the 2D slice data in the last 2 dimensions of the 3D Tensors. 
        # Your tensor needs to be of shape [1, patch_size, patch_size]
        # Don't forget that you need to put a Torch Tensor into your dictionary element's value
        # Hint: your 3D data sits in self.data variable, the id of the 3D volume from data array
        # and the slice number are in the slc variable. 
        # Hint2: You can use None notation like so: arr[None, :] to add size-1 
        # dimension to a Numpy array
        # <YOUR CODE GOES HERE>

        # 'slice' is a list of tuples with the format (volume_ix, slice_ix). Get these values.
        volume_ix, slice_ix = slc

        # 'data' is a np-array. Each element is a dict with keys=['image', 'seg', 'filename']
        # The volume of images is in 'image', and the volume of labels is in 'seg', so we get a
        # single item with the indices
        image = self.data[volume_ix]['image'][slice_ix]
        label = self.data[volume_ix]['seg'][slice_ix]

        # Transform these data from numpy arrays to torch tensors and leave them in the desired format
        # Here tensors could be passed to GPU, but I'll do it later when checking if cuda is available
        sample['image'] = torch.from_numpy(image).unsqueeze(0)
        sample['seg'] = torch.from_numpy(label).unsqueeze(0)

        return sample

    def __len__(self):
        """
        This method is called by PyTorch DataLoader class to return number of samples in the dataset

        Returns:
            int
        """
        return len(self.slices)