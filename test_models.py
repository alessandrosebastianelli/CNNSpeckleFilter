import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np
from findpeaks import findpeaks
import findpeaks
import bm3d

def preprocessing_int2net(img):
    return img.abs().log()/2

def postprocessing_net2int(img):
    return (2*img).exp()

def single2tensor4(img):
      return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)

def uint2single(img):
    return np.float32(img/255.)


def test_DNCNN(img):
    sys.path.append('DPIR')
    sys.path.append('DPIR/utils')
    sys.path.append('DPIR/models')

    from network_unet import UNetRes as net
    import torch
    

    device = 'cpu'
    idcnn = net(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    idcnn.load_state_dict(torch.load('other_weights/drunet_gray.pth'), strict=True)
    idcnn.eval()
    for k, v in idcnn.named_parameters():
        v.requires_grad = False
    idcnn = idcnn.to(device)

    img_L = single2tensor4(uint2single(img[0:64,0:64,...]))
    img_L = torch.cat((img_L, torch.FloatTensor([15/255.]).repeat(1, 1, img_L.shape[2], img_L.shape[3])), dim=1)
    img_L = img_L.to(device)
    res = idcnn(torch.FloatTensor(img_L)).detach().numpy()

    sys.path.remove('DPIR')
    sys.path.remove('DPIR/utils')
    sys.path.remove('DPIR/models')

    return res

def test_IDCNN(img):
    sys.path.append('IDCNN')
    matplotlib.image.imsave('test/test.png', img, cmap='gray') 
    os.system("python IDCNN/inference.py --test_file test/test.png --checkpoint_dir IDCNN/results/checkpoint_impulses_bsd500_41/ --save_dir test/ --phase inference") 
    IDCNN = plt.imread('test/denoised_CNN_test.png')
    IDCNN = IDCNN[...,1]
    os.remove('test/test.png')
    os.remove('test/denoised_CNN_test.png')
    os.remove('test/detected_impulses_CNN_test.png')
    sys.path.remove('IDCNN')

    return IDCNN

def test_classic(img):
    # filters parameters
    # window size
    winsize = 3
    # damping factor for frost
    k_value1 = 2.0
    # damping factor for lee enhanced
    k_value2 = 1.0
    # coefficient of variation of noise
    cu_value = 0.25
    # coefficient of variation for lee enhanced of noise
    cu_lee_enhanced = 0.523
    # max coefficient of variation for lee enhanced
    cmax_value = 1.73
    # Classic
    image_lee = findpeaks.lee_filter(img, win_size=winsize, cu=cu_value)
    image_lee_enhanced = findpeaks.lee_enhanced_filter(img, win_size=winsize, k=k_value2, cu=cu_lee_enhanced, cmax=cmax_value)
    image_kuan = findpeaks.kuan_filter(img, win_size=winsize, cu=cu_value)
    image_frost = findpeaks.frost_filter(img, damping_factor=k_value1, win_size=winsize)
    image_mean = findpeaks.mean_filter(img.copy(), win_size=winsize)
    image_median = findpeaks.median_filter(img, win_size=winsize)
    img_fastnl = findpeaks.stats.denoise(img, method='fastnl', window=winsize)
    img_bilateral = findpeaks.stats.denoise(img, method='bilateral', window=winsize)

    return image_lee, image_lee_enhanced, image_kuan, image_frost, image_mean, image_median, img_fastnl, img_bilateral

def test_BM3D(img):
    img_bm3d = bm3d.bm3d(img, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)#bm3d.bm3d(img, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    return img_bm3d

def test_SARCNN(img):
    sys.path.append('SAR-CNN')
    sys.path.append('SAR-CNN/models')
    import DnCNN as DnCNN

    with open('other_weights/SAR_CNN_e50.pkl', "rb") as fid:
        dncnn_opt = dict(**pickle.load(fid).dncnn)
        dncnn_opt["residual"] = True
    sarcnn = DnCNN.DnCNN(1, 1, **dncnn_opt)
    sarcnn.load_state_dict(torch.load('other_weights/SAR_CNN_e50.t7', map_location=torch.device('cpu'))['net'])

    # SARCNN
    with torch.no_grad():
        #target_intensity = torch.from_numpy((1+speckle[0:64,0:64,...]).astype(np.float32))[None, None, :, :]
        #randomStream = np.random.RandomState(32)
        #noise_int    = randomStream.gamma(size=(64,64), shape=1.0, scale=1.0)
        #noise_int  = torch.from_numpy(noise_int.astype(np.float32))[None, None, :, :]

        noisy_int  = torch.from_numpy(((1/256.0)+img[0:64, 0:64, 0]).astype(np.float32))[None, None, :, :]
        #noisy_int = preprocessing_int2net(noisy_int)
        pred_int = sarcnn(noisy_int)

        #pred_int = postprocessing_net2int(pred_int)
        pred_int = pred_int.detach().numpy()    
        #pred_int = (pred_int - pred_int.min())/((pred_int.max() - pred_int.min()) + 1e-6)
    
    sys.path.remove('SAR-CNN')
    sys.path.remove('SAR-CNN/models')

    return pred_int

def test_CNNNLM(img):
    sys.path.append('CNN-NLM')
    sys.path.append('CNN-NLM/utils')

    import models.DnCNN as DnCNN
    import pickle
    import torch

    with open("CNN-NLM/weights/sar_sync/SAR_CNN_e50.pkl", "rb") as fid:
        dncnn_opt = dict(**pickle.load(fid).dncnn)
        dncnn_opt["residual"] = True
    net = DnCNN.DnCNN(1, 1, **dncnn_opt)
    net.load_state_dict(torch.load('CNN-NLM/weights/sar_sync/SAR_CNN_e50.t7', map_location=torch.device('cpu'))['net'])
    pad = 0
    
    with torch.no_grad():
        noise_int  = torch.from_numpy(((1/255)+img[:64, :64, 0]).astype(np.float32))[None, None, :, :]  
        #noise_int =  preprocessing_int2net(noise_int)
        pred_int = net(noise_int)
        #pred_int = postprocessing_net2int(pred_int)

    sys.path.remove('CNN-NLM')
    sys.path.remove('CNN-NLM/utils')

    return pred_int.numpy() 

def test_speckle2void(img):
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    sys.path.append('speckle2void')
    sys.path.append('speckle2void/libraries')
    #sys.path.insert(0, './libraries')

    img2 = np.zeros((1, img.shape[0], img.shape[1],1))
    img2[0,...] = img

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from Speckle2Void import Speckle2V
    tf.reset_default_graph()
    batch_size=16

    dir_train = None
    dir_test = None

    file_checkpoint = 'speckle2void/s2v_checkpoint/model.ckpt-299999'#None for the latest checkpoint

    model = Speckle2V(dir_train,
                    dir_test,
                    file_checkpoint,
                    batch_size=batch_size,
                    patch_size=64,
                    model_name='speckle2void',
                    lr=1e-04, 
                    steps_per_epoch=2000,
                    k_penalty_tv=5e-05,
                    shift_list=[3,1],
                    prob = [0.9,0.1],
                    clip=1,
                    norm=1,
                    L_noise=1)   

    model.build_inference()
    model.load_weights()
    batch_pred = model.predict(img2)

    sys.path.remove('speckle2void')
    sys.path.remove('speckle2void/libraries')

    return batch_pred[0,...]

def test_SAR2SAR(img):
    sys.path.append('SAR2SAR-GRD-test')

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from model import denoiser
    tf.reset_default_graph()

    img2 = np.zeros((1, img.shape[0], img.shape[1],1))
    img2[0,...] = img


    with tf.Session() as sess:
        model = denoiser(sess)
        model.load('SAR2SAR-GRD-test/checkpoint')
        Y_ = tf.placeholder(tf.float32, [None, None, None, 1],
                                 name='clean_image')
        pred = sess.run([model.Y], feed_dict={model.Y_: img2})

    sys.path.remove('SAR2SAR-GRD-test')

    return pred[0][0,...]

