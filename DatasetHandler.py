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
            s1 = src.read()
            s1 = np.transpose(s1)

        return s1
    
    def __normalize(self, s1):
        def reject_outliers_2(data, m=5):
            d = np.abs(data - np.median(data))
            mdev = np.median(d)
            s = d / (mdev if mdev else 1.)
            return data[s < m]  

        d = reject_outliers_2(s1.flatten(), m=6.)

        s1 = (s1 -  np.min(d))/(np.max(d) -  np.min(d))
        #s1 = (s1 -  np.min(s1))/(np.max(s1) -  np.min(s1))
        s1 = np.clip(s1, 0.0, 1.0)

        return s1.astype(np.float)

    def __add_speckle(self, s1, mean = 0, sigma = 0.25):
        #x = np.random.normal(mean, sigma**2, s1.shape)
        #y = np.random.normal(mean, sigma**2, s1.shape)
        #noise = np.sqrt(x**2, y**2)
        noise = np.random.rayleigh(sigma, s1.shape[0]*s1.shape[1]).reshape(s1.shape)
        noise = noise
        s1 = (s1*noise)

        #s1 = np.clip(s1, 0.0, 1.0)
        return s1.astype(np.float), noise

    def data_loader(self, paths, batch_size, img_shape, out_noise=False):
        batch_speckle = np.zeros((batch_size, img_shape[0], img_shape[1], 1))
        batch_clean = np.zeros((batch_size, img_shape[0], img_shape[1], 1))

        if out_noise:
          batch_noise = np.zeros((batch_size, img_shape[0], img_shape[1], 1))

        indexes = random.sample(range(len(paths)-1), len(paths)-1)
        counter = 0

        while True:
            #if counter > len(paths)-(batch_size+1):
            #    indexes = random.sample(range(len(paths)-1), len(paths)-1)
            #    counter = 0
            #    print('Restart')

            for i in range(batch_size):
                idx = random.randint(0, len(paths) - 1)
                s1 = self.__load(paths[idx])
                #s1 = self.__load(paths[indexes[i+counter]])
                #s1 = self.__normalize(s1)
                s1 = s1[0:img_shape[0], 0:img_shape[1],:]
                s1 = self.__normalize(s1)

                batch_clean[i,0:img_shape[0], 0:img_shape[1], :] = s1 #self.__normalize(s1)

                if out_noise:
                  batch_speckle[i,0:img_shape[0], 0:img_shape[1], :], batch_noise[i,0:img_shape[0], 0:img_shape[1], :] = self.__add_speckle(s1)
                else:  
                  batch_speckle[i,0:img_shape[0], 0:img_shape[1], :], _ = self.__add_speckle(s1)

                #batch_speckle[i,0:img_shape[0], 0:img_shape[1], :] = batch_speckle[i,0:img_shape[0], 0:img_shape[1], :]
                batch_speckle[i,0:img_shape[0], 0:img_shape[1], :] = self.__normalize(batch_speckle[i,0:img_shape[0], 0:img_shape[1], :])
                counter += 1
            if out_noise:
              yield batch_speckle, batch_clean, batch_noise
            else:
              yield batch_speckle, batch_clean

    def data_loader_v2(self, paths, img_shape):
        batch_size = len(paths)
        batch_speckle = np.zeros((batch_size, img_shape[0], img_shape[1], 1))
        batch_clean = np.zeros((batch_size, img_shape[0], img_shape[1], 1))

        for i in range(batch_size):
            s1 = self.__load(paths[i])
            #s1 = self.__load(paths[indexes[i+counter]])
            #s1 = self.__normalize(s1)
            s1 = s1[0:img_shape[0], 0:img_shape[1],:]
            s1 = self.__normalize(s1)

            batch_clean[i,0:img_shape[0], 0:img_shape[1], :] = s1 #self.__normalize(s1) 
            batch_speckle[i,0:img_shape[0], 0:img_shape[1], :], _ = self.__add_speckle(s1)

            #batch_speckle[i,0:img_shape[0], 0:img_shape[1], :] = batch_speckle[i,0:img_shape[0], 0:img_shape[1], :]
            batch_speckle[i,0:img_shape[0], 0:img_shape[1], :] = self.__normalize(batch_speckle[i,0:img_shape[0], 0:img_shape[1], :])
         
        return batch_speckle, batch_clean