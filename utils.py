import matplotlib.pyplot as plt
import numpy as np

def plot_dataset(batch_speckle, batch_clean):
    n = batch_speckle.shape[0]

    for i in range(n):
        fig, axes = plt.subplots(
        nrows = 2,
        ncols = 2, 
        figsize = (8, 8))

        axes[0,0].imshow(batch_speckle[i,...,0], cmap='gray')
        axes[0,0].set_title('Input with speckle')
        axes[0,1].imshow(batch_clean[i,...,0], cmap='gray')
        axes[0,1].set_title('Ground truth')

        axes[1,0].hist(batch_speckle[i,...,0].flatten(), bins=20, histtype='step')
        axes[1,0].set_title('Input with speckle')
        axes[1,1].hist(batch_clean[i,...,0].flatten(), bins=20, histtype='step')
        axes[1,1].set_title('Ground truth')

        plt.show()
        plt.close()

def plot_history(history):
    fig, ax = plt.subplots(
        nrows = 2,
        ncols = 1, 
        figsize = (15,10))

    ax[0].plot(history.history['loss'], '-*', label='Training Loss')
    ax[0].plot(history.history['val_loss'], '-o', label='Validation Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('MAE')
    ax[0].set_title('Training VS Validation MAE')

    ax[1].plot(history.history['mse'], '-*', label='Training Loss')
    ax[1].plot(history.history['val_mse'], '-o', label='Validation Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('MSE')
    ax[1].set_title('Training VS Validation MSE')
    
    ax[0].legend()
    ax[0].grid()
    ax[1].legend()
    ax[1].grid()
    plt.show()

def plot_model_results(batch_speckle, batch_clean, batch_pred):
    n = batch_speckle.shape[0]


    for i in range(n):
        fig, axes = plt.subplots(
        nrows = 2,
        ncols = 4, 
        figsize = (16, 8))

        axes[0,0].imshow(batch_speckle[i,...,0], cmap='gray')
        axes[0,0].set_title('Input with speckle')
        axes[0,1].imshow(batch_clean[i,...,0], cmap='gray')
        axes[0,1].set_title('Ground truth')
        axes[0,2].imshow(batch_pred[i,...,0], cmap='gray')
        axes[0,2].set_title('Model Prediction')
        diff = np.abs(batch_pred[i,...,0] - batch_clean[i,...,0])
        axes[0,3].imshow(diff, vmin=np.min(diff), vmax=np.max(diff), cmap='gray')
        axes[0,3].set_title('|Model Prediction - Ground Truth|')

        axes[1,0].hist(batch_speckle[i,...,0].flatten(), bins=20, histtype='step')
        axes[1,0].set_title('Input with speckle')
        axes[1,1].hist(batch_clean[i,...,0].flatten(), bins=20, histtype='step')
        axes[1,1].set_title('Ground truth')
        axes[1,2].hist(batch_pred[i,...,0].flatten(), bins=20, histtype='step')
        axes[1,2].set_title('Model Prediction')
        axes[1,3].hist(diff.flatten(), bins=20, histtype='step')
        axes[1,3].set_title('|Model Prediction - Ground Truth|')
    
        plt.show()
        plt.close()