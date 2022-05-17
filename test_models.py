from findpeaks import findpeaks
import findpeaks
import bm3d


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
    img_bm3d = bm3d.bm3d(img, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    return img_bm3d