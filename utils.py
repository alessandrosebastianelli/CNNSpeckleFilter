import matplotlib.pyplot as plt


def plot_dataset(batch_speckle, batch_clean):
    n = batch_speckle.shape[0]
    fig, axes = plt.subplots(
        nrows = 2*n,
        ncols = 2, 
        figsize = (8, 2*n*4))

    for i in range(n):
        axes[2*i,0].imshow(batch_speckle[i,...,0], cmap='gray')
        axes[2*i,0].set_title('Input with speckle')
        axes[2*i,1].imshow(batch_clean[i,...,0], cmap='gray')
        axes[2*i,1].set_title('Ground truth')

        axes[2*i+1,0].hist(batch_speckle[i,...,0].flatten(), bins=20, histtype='step')
        axes[2*i+1,0].set_title('Input with speckle')
        axes[2*i+1,1].hist(batch_clean[i,...,0].flatten(), bins=20, histtype='step')
        axes[2*i+1,1].set_title('Ground truth')

    plt.show()

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