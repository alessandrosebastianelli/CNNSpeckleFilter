import numpy as np
import rasterio
import random
import glob
import os


class DatasetHandler():

    def __init__(self, root):
        self.train_paths = glob.glob(
            os.path.join(os.path.join(
                root,
                "training"
            ), "*")
        )
        self.val_paths = glob.glob(
            os.path.join(os.path.join(
                root,
                "validation"
            ), "*")
        )

    def __load(self, path):
        with rasterio.open(path) as src:
            s1 = src.read()
            s1 = np.transpose(s1)

        return s1
    
    def __normalize(self, s1):
        minval = np.min(s1)
        s1 = (s1 - minval)/(np.max(s1) - minval)
        s1 = np.clip(s1, 0.0, 1.0)

        return s1.astype(np.float)

    def __add_speckled(self, s1, mean = 0, sigma = 0.3):
        x = np.random.normal(mean, sigma**0.5, s1.shape)
        y = np.random.normal(mean, sigma**0.5, s1.shape)
        noise = np.sqrt(x**2, y**2)
        s1 = s1*noise
        s1 = np.clip(s1, 0.0, 1.0)

        return s1.astype(np.float)

    def data_loader(self, paths, batch_size, img_shape):
        batch_speckle = np.zeros((batch_size, img_shape[0], img_shape[1], 1))
        batch_clean = np.zeros((batch_size, img_shape[0], img_shape[1], 1))

        indexes = random.sample(range(len(paths)), len(paths))
        counter = 0

        while True:
            if counter > len(paths)-batch_size:
                indexes = random.sample(range(len(paths)), len(paths))
                counter = 0

            for i in range(batch_size):
                s1 = self.__load(paths[indexes[i+counter]])
                s1 = self.__normalize(s1)

                batch_clean[i,...] = s1
                batch_speckle[i,...] = self.__add_speckled(s1)
                counter += 1

            yield batch_speckle, batch_clean