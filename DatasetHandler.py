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

    def load(self, path):
        with rasterio.open(path) as src:
            intensity = src.read()
            intensity = np.moveaxis(intensity,0,-1)

        return intensity
    
    def reject_outliers(self, s1):
        p = np.percentile(s1, 90)
        s1 = np.clip(s1, 0.0, p)

        return s1.astype(np.float)
    
    def min_max(self, s1):
        return (s1 - s1.min()) / (s1.max() - s1.min()).astype(np.float)

    def add_speckle(self, s1, looks = 4):
        # Numpy Gamma Distribution is defined in the shape-scale form
        # Mean 1 Var 1/looks
        gamma_shape = looks
        gamma_scale = 1/looks

        noise = np.random.gamma(gamma_shape, 
                                gamma_scale, 
                                s1.shape[0]*s1.shape[1]).reshape(s1.shape)
        s1 = s1*noise

        return s1.astype(np.float), noise.astype(np.float)

    def data_loader(self, paths, batch_size, img_shape, out_noise=False):
        batch_speckle = np.zeros((batch_size, img_shape[0], img_shape[1], 1))
        batch_clean = np.zeros((batch_size, img_shape[0], img_shape[1], 1))

        if out_noise:
            batch_noise = np.zeros((batch_size, img_shape[0], img_shape[1], 1))

        indexes = random.sample(range(len(paths)-1), len(paths)-1)
        counter = 0

        while True:
            for i in range(batch_size):
                # Load Image
                idx = random.randint(0, len(paths) - 1)
                s1 = self.load(paths[idx])
                s1 = s1[0:img_shape[0], 0:img_shape[1],:]
                                
                # Adding speckle
                if out_noise:
                    s1_speckle, batch_noise[i,0:img_shape[0], 0:img_shape[1], :] = self.add_speckle(s1)
                else:  
                    s1_speckle, _ = self.add_speckle(s1)

                # Filling noisy inputs - normalized between 0-1
                s1_speckle = self.reject_outliers(s1_speckle)
                s1_speckle = self.min_max(s1_speckle)
                batch_speckle[i,0:img_shape[0], 0:img_shape[1], :] = s1_speckle
                
                # Filling ground truths - normalized between 0-1
                s1 = self.reject_outliers(s1)
                s1 = self.min_max(s1)
                batch_clean[i,0:img_shape[0], 0:img_shape[1], :] = s1
                
                counter += 1
                
                
            if out_noise:
                yield np.clip(batch_speckle,0.0,1.0), np.clip(batch_clean,0.0,1.0), batch_noise
            else:
                yield np.clip(batch_speckle,0.0,1.0), np.clip(batch_clean,0.0,1.0)