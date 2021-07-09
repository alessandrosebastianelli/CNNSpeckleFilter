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
        self.test_paths = glob.glob(
            os.path.join(os.path.join(
                root,
                "testing"
            ), "*")
        )

    def __load(self, path):
        with rasterio.open(path) as src:
            intensity = src.read()
            intensity = np.transpose(intensity)
        return intensity.astype(np.float)
    
    def __normalize(self, s1, MAX, MIN):
        def reject_outliers_2(data, m=5):
            d = np.abs(data - np.median(data))
            mdev = np.median(d)
            s = d / (mdev if mdev else 1.)
            return data[s < m]  

        d = reject_outliers_2(s1.flatten(), m=5.)
     
        s1_n = (s1 -  np.min(d))/(np.max(d) -  np.min(d))

        s1_n = np.clip(s1_n, 0.0, 1.0)

        return s1_n.astype(np.float)

    def __add_speckle(self, s1, looks = 4):
        # Numpy Gamma Distribution is defined in the shape-scale form
        # Mean 1 Var 1/looks
        gamma_shape = looks
        gamma_scale = 1/looks

        noise = np.random.gamma(gamma_shape, 
                                gamma_scale, 
                                s1.shape[0]*s1.shape[1]).reshape(s1.shape)
        s1 = s1*noise

        return s1.astype(np.float), noise

    def data_loader(self, paths, batch_size, img_shape, MAX, MIN, out_noise=False):
        batch_speckle = np.zeros((batch_size, img_shape[0], img_shape[1], 1))
        batch_clean = np.zeros((batch_size, img_shape[0], img_shape[1], 1))

        if out_noise:
          batch_noise = np.zeros((batch_size, img_shape[0], img_shape[1], 1))

        indexes = random.sample(range(len(paths)-1), len(paths)-1)
        counter = 0

        while True:
            for i in range(batch_size):
                idx = random.randint(0, len(paths) - 1)
                s1 = self.__load(paths[idx])
                s1 = s1[0:img_shape[0], 0:img_shape[1],:]
                #s1 = self.__normalize(s1)

                batch_clean[i,0:img_shape[0], 0:img_shape[1], :] = self.__normalize(s1, MAX, MIN)

                if out_noise:
                  batch_speckle[i,0:img_shape[0], 0:img_shape[1], :], batch_noise[i,0:img_shape[0], 0:img_shape[1], :] = self.__add_speckle(s1)
                else:  
                  batch_speckle[i,0:img_shape[0], 0:img_shape[1], :], _ = self.__add_speckle(s1)

                batch_speckle[i,0:img_shape[0], 0:img_shape[1], :] = self.__normalize(batch_speckle[i,0:img_shape[0], 0:img_shape[1], :], MAX, MIN)
                counter += 1
            if out_noise:
              yield batch_speckle, batch_clean, batch_noise
            else:
              yield batch_speckle, batch_clean

    def data_loader_v2(self, paths, img_shape, MAX, MIN):
        batch_size = len(paths)
        batch_speckle = np.zeros((batch_size, img_shape[0], img_shape[1], 1))
        batch_clean = np.zeros((batch_size, img_shape[0], img_shape[1], 1))

        for i in range(batch_size):
            s1 = self.__load(paths[i])
            #s1 = self.__load(paths[indexes[i+counter]])
            #s1 = self.__normalize(s1)
            s1 = s1[0:img_shape[0], 0:img_shape[1],:]
            #s1 = self.__normalize(s1)

            batch_clean[i,0:img_shape[0], 0:img_shape[1], :] = self.__normalize(s1, MAX, MIN) 
            batch_speckle[i,0:img_shape[0], 0:img_shape[1], :], _ = self.__add_speckle(s1)

            #batch_speckle[i,0:img_shape[0], 0:img_shape[1], :] = batch_speckle[i,0:img_shape[0], 0:img_shape[1], :]
            batch_speckle[i,0:img_shape[0], 0:img_shape[1], :] = self.__normalize(batch_speckle[i,0:img_shape[0], 0:img_shape[1], :], MAX, MIN)
         
        return batch_speckle, batch_clean

    def getstats(self, paths):
        batch_size = len(paths)

        maxs = []
        mins = []

        for i in range(batch_size):
            s1 = self.__load(paths[i])
            maxs.append(np.max(s1))
            mins.append(np.min(s1))

            print('\r Image ' + str(i) + ' of ' + str(batch_size), end='\t')

        maxs = np.array(maxs)
        mins = np.array(mins)

        return np.nanmax(maxs), np.nanmin(mins)