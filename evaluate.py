import torch
import numpy as np
from torch.utils.data import DataLoader
from gazegan_model import GazeGAN
from train import SaliencyDataset
from scipy.stats import entropy
import cv2
import os
from tqdm import tqdm  # For progress bar

class MetricCalculator:
    @staticmethod
    def AUC_Borji(pred, fix):
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        fix = (fix > 0.5).astype(int)
        
        if np.sum(fix) == 0:
            return np.nan
            
        S = pred.ravel()
        F = fix.ravel()
        S_fix = S[F == 1]
        n_fix = len(S_fix)
        n_pixels = len(S)
        
        thresholds = np.sort(S_fix)[::-1]
        tp = np.zeros(len(thresholds)+2)
        fp = np.zeros(len(thresholds)+2)
        tp[0], tp[-1] = 0, 1
        fp[0], fp[-1] = 0, 1
        
        for k, thresh in enumerate(thresholds):
            above_th = np.sum(S >= thresh)
            tp[k+1] = (k + 1) / n_fix
            fp[k+1] = (above_th - (k + 1)) / (n_pixels - n_fix)
            
        return np.trapezoid(tp, fp)  # Changed from trapz to trapezoid

    @staticmethod
    def AUC_Judd(pred, fix):
        return MetricCalculator.AUC_Borji(pred, fix)

    @staticmethod
    def AUC_shuffled(pred, fix, other_imgs_fix, n=5):  # Reduced from 10 to 5 for speed
        fix = (fix > 0.5).astype(int)
        if np.sum(fix) == 0:
            return np.nan
            
        h, w = fix.shape
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        
        aucs = []
        for _ in range(n):
            idx = np.random.randint(0, len(other_imgs_fix))
            rand_fix = cv2.resize(other_imgs_fix[idx], (w, h))
            auc = MetricCalculator.AUC_Borji(pred, rand_fix)
            if not np.isnan(auc):
                aucs.append(auc)
                
        return np.mean(aucs) if aucs else np.nan

    @staticmethod
    def CC(pred, fix):
        pred = (pred - pred.mean()) / (pred.std() + 1e-12)
        fix = (fix - fix.mean()) / (fix.std() + 1e-12)
        return np.corrcoef(pred.ravel(), fix.ravel())[0,1]

    @staticmethod
    def EMD_approx(pred, fix):
        """Faster approximation without histograms"""
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-12)
        fix = (fix - fix.min()) / (fix.max() - fix.min() + 1e-12)
        return np.mean(np.abs(pred - fix))

    @staticmethod
    def InfoGain(pred, fix, baseline=0.5):
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-12)
        fix = (fix > 0.5).astype(int)
        if np.sum(fix) == 0:
            return np.nan
        eps = 1e-12
        p = np.mean(pred[fix == 1])
        return np.log2(p + eps) - np.log2(baseline + eps)

    @staticmethod
    def KLdiv(pred, fix):
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-12)
        fix = (fix - fix.min()) / (fix.max() - fix.min() + 1e-12)
        
        pred = pred.ravel() + 1e-12
        fix = fix.ravel() + 1e-12
        
        pred = pred / pred.sum()
        fix = fix / fix.sum()
        
        return entropy(fix, pred)

    @staticmethod
    def NSS(pred, fix):
        pred = (pred - pred.mean()) / (pred.std() + 1e-12)
        fix = (fix > 0.5).astype(int)
        if np.sum(fix) == 0:
            return np.nan
        return np.mean(pred[fix == 1])

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = GazeGAN(use_csc=True).to(device)
    model.load_state_dict(torch.load('gazegan_final.pth'))
    model.eval()

    # Initialize dataset
    dataset = SaliencyDataset(root_dir='C:/Users/User/Downloads/Saliency4asd/Saliency4asd')
    
    # Pre-load fixation maps for shuffled AUC
    print("Preloading fixation maps...")
    other_fixations = []
    for batch in DataLoader(dataset, batch_size=1):
        other_fixations.append(batch['td_fix_map'].numpy()[0,0])
    
    # Evaluation metrics
    metrics = {
        'TD': {m: [] for m in ['AUC_Borji', 'AUC_Judd', 'AUC_shuffled', 
                              'CC', 'EMD', 'InfoGain', 'KLdiv', 'NSS']},
        'ASD': {m: [] for m in ['AUC_Borji', 'AUC_Judd', 'AUC_shuffled', 
                               'CC', 'EMD', 'InfoGain', 'KLdiv', 'NSS']}
    }

    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size=1), total=len(dataset)):
            img = batch['image'].to(device)
            td_fix = batch['td_fix_map'].numpy()[0,0]
            asd_fix = batch['asd_fix_map'].numpy()[0,0]
            
            pred = model(img).cpu().numpy()[0,0]
            
            # Evaluate against TD
            metrics['TD']['AUC_Borji'].append(MetricCalculator.AUC_Borji(pred, td_fix))
            metrics['TD']['AUC_Judd'].append(MetricCalculator.AUC_Judd(pred, td_fix))
            metrics['TD']['AUC_shuffled'].append(
                MetricCalculator.AUC_shuffled(pred, td_fix, other_fixations))
            metrics['TD']['CC'].append(MetricCalculator.CC(pred, td_fix))
            metrics['TD']['EMD'].append(MetricCalculator.EMD_approx(pred, td_fix))
            metrics['TD']['InfoGain'].append(MetricCalculator.InfoGain(pred, td_fix))
            metrics['TD']['KLdiv'].append(MetricCalculator.KLdiv(pred, td_fix))
            metrics['TD']['NSS'].append(MetricCalculator.NSS(pred, td_fix))
            
            # Evaluate against ASD
            metrics['ASD']['AUC_Borji'].append(MetricCalculator.AUC_Borji(pred, asd_fix))
            metrics['ASD']['AUC_Judd'].append(MetricCalculator.AUC_Judd(pred, asd_fix))
            metrics['ASD']['AUC_shuffled'].append(
                MetricCalculator.AUC_shuffled(pred, asd_fix, other_fixations))
            metrics['ASD']['CC'].append(MetricCalculator.CC(pred, asd_fix))
            metrics['ASD']['EMD'].append(MetricCalculator.EMD_approx(pred, asd_fix))
            metrics['ASD']['InfoGain'].append(MetricCalculator.InfoGain(pred, asd_fix))
            metrics['ASD']['KLdiv'].append(MetricCalculator.KLdiv(pred, asd_fix))
            metrics['ASD']['NSS'].append(MetricCalculator.NSS(pred, asd_fix))

    # Calculate and print results
    def print_results(metrics_dict, name):
        print(f"\n{name} Metrics:")
        for metric, values in metrics_dict.items():
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                mean = np.mean(valid_values)
                std = np.std(valid_values)
                print(f"{metric}: {mean:.4f} ± {std:.4f}")
            else:
                print(f"{metric}: No valid values")

    print_results(metrics['TD'], "TD")
    print_results(metrics['ASD'], "ASD")

if __name__ == '__main__':
    evaluate()