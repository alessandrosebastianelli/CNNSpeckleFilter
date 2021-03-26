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
        nrows = 1,
        ncols = 1, 
        figsize = (10,4)

    ax.plot(history.history['loss'], '-*', label='Training Loss')
    ax.plot(history.history['val_loss'], '-.', label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MAE')
    
    plt.show()