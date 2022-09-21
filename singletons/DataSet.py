import os
import collections
import numpy as np

dSpritesDataset = collections.namedtuple('dSpritesDataset', field_names=['images', 's_sizes', 's_dim', 's_bases'])


#
# Singleton to access datasets.
#
class DataSet:

    instance = {}

    @staticmethod
    def get(images_archive):
        """
        Getter
        :param images_archive: the file in which the dataset is stored
        :return: an object containing the dSprite dataset
        """
        file_name = os.path.basename(images_archive)
        if file_name not in DataSet.instance.keys():
            loaders = {
                "dsprites.npz": DataSet.load_sprites_dataset,
                "pong-v5.npz": DataSet.load_openai_dataset,
                "boxing-v5.npz": DataSet.load_openai_dataset
            }
            DataSet.instance[file_name] = loaders[file_name](images_archive)
        return DataSet.instance[file_name]

    @staticmethod
    def load_sprites_dataset(images_archive):
        dataset = np.load(images_archive, allow_pickle=True, encoding='latin1')
        images = dataset['imgs'].reshape(-1, 64, 64, 1)
        metadata = dataset['metadata'][()]
        s_sizes = metadata['latents_sizes']  # [1 3 6 40 32 32]
        s_dim = s_sizes.size
        s_bases = np.concatenate((metadata['latents_sizes'][::-1].cumprod()[::-1][1:], np.array([1, ])))
        s_bases = np.squeeze(s_bases)  # self.s_bases = [737280 245760  40960 1024 32]
        return dSpritesDataset(images, s_sizes, s_dim, s_bases)

    @staticmethod
    def load_openai_dataset(images_archive):
        dataset = np.load(images_archive, allow_pickle=True, encoding='latin1')
        return dSpritesDataset(dataset['images'].reshape(-1, 64, 64, 1), None, None, None)

