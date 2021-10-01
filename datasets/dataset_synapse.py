import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class JigsawTransformation(object):
    def __init__(self, K, Q, output_size):
        self.K = K
        self.Q = Q
        self.permutations = self.__retrive_permutations(K)
        self.output_size = output_size
        assert output_size % 3 == 0, "Output size should have a factor of 3"

    def __call__(self, sample):
        image = sample['image']

        if image.size[0] != self.output_size:
            image = transforms.Resize((self.output_size, self.output_size))(image)

        jigsaw_images = np.zeros((self.Q, self.output_size, self.output_size))
        jigsaw_labels = np.zeros((self.Q, 1))
        piece_size = float(image.size[0]) / 3

        for q in range(self.Q):
            perm_index = np.random.randint(len(self.permutations))
            permutation = self.permutations[perm_index]
            for n in range(9):
                i_start = n / 3 * piece_size
                j_start = n % 3 * piece_size
                i_start_new = permutation[n] / 3 * piece_size
                j_start_new = permutation[n] % 3 * piece_size
                jigsaw_images[q, i_start_new : i_start_new + piece_size, j_start_new : j_start_new + piece_size] = \
                    image[i_start : i_start + piece_size, j_start : j_start + piece_size]
            jigsaw_labels[q] = perm_index

        sample['jigsaw_images'] = jigsaw_images
        sample['jigsaw_labels'] = jigsaw_labels
        return sample

    def __retrive_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).read().splitlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]

        if self.split == "train":
            path = os.path.join(self.data_dir, name+'.npz')
            data = np.load(path)
            image, label = data['image'], data['label']
        else:
            path = os.path.join(self.data_dir, name+'.npy.h5')
            data = h5py.File(path)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label, 'case_name': self.sample_list[idx]}

        if self.transform:
            sample = self.transform(sample)
        
        return sample
