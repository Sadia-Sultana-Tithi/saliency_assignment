import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy
import cv2

def AUC_Judd(saliency_map, fixation_map, jitter=True):
    saliency_map = np.array(saliency_map, copy=True)
    fixation_map = np.array(fixation_map, copy=True) > 0.5
    
    if not np.any(fixation_map):
        return np.nan
    
    if jitter:
        saliency_map += np.random.rand(*saliency_map.shape) * 1e-12
    
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    
    S = saliency_map.ravel()
    F = fixation_map.ravel()
    
    S_fix = S[F]
    n_fix = len(S_fix)
    n_pixels = len(S)
    
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds)+2)
    fp = np.zeros(len(thresholds)+2)
    tp[0] = 0; tp[-1] = 1
    fp[0] = 0; fp[-1] = 1
    
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh)
        tp[k+1] = (k + 1) / float(n_fix)
        fp[k+1] = (above_th - (k + 1)) / float(n_pixels - n_fix)
    
    return np.trapz(tp, fp)

def NSS(saliency_map, fixation_map):
    saliency_map = np.array(saliency_map, copy=True)
    fixation_map = np.array(fixation_map, copy=True) > 0.5
    
    if not np.any(fixation_map):
        return np.nan
    
    saliency_map = (saliency_map - saliency_map.mean()) / saliency_map.std()
    return np.mean(saliency_map[fixation_map])

def CC(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=True).ravel()
    map2 = np.array(saliency_map2, copy=True).ravel()
    
    if np.isnan(map1).any() or np.isnan(map2).any():
        return np.nan
    
    map1 = (map1 - map1.mean()) / map1.std()
    map2 = (map2 - map2.mean()) / map2.std()
    
    return np.corrcoef(map1, map2)[0,1]

def similarity(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=True)
    map2 = np.array(saliency_map2, copy=True)
    
    if np.isnan(map1).any() or np.isnan(map2).any():
        return np.nan
    
    map1 = (map1 - map1.min()) / (map1.max() - map1.min())
    map2 = (map2 - map2.min()) / (map2.max() - map2.min())
    
    map1 = map1 / map1.sum()
    map2 = map2 / map2.sum()
    
    return np.minimum(map1, map2).sum()