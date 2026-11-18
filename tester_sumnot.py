import argparse
import os
import glob
import csv
import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pyiqa # Import for NIQE and PIQE

# Import your model and dataset
from model_fvmot import * # Note: imported from model_fvmot to match training file
from utils.universal_dataset import MFMRI_TestDataset


# =========================================================================
# Added: Dataset specifically for real data (No GT)
# =========================================================================
class InferenceDataset(Dataset):
    def __init__(self, folder_path):
        super().__init__()
        self.toTensor = ToTensor()
        exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        self.files = []
        for ext in exts:
            self.files.extend(glob.glob(os.path.join(folder_path, ext)))
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        fname = os.path.basename(img_path)
        img = Image.open(img_path).convert('RGB')
        return fname, self.toTensor(img)

# =========================================================================
# Added: Pure inference function (Calculates NIQE/PIQE, saves DEG and OUT)
# =========================================================================
def inference_only(model, val_loader, opt, save_dir, iqa_metrics):
    model.eval()
    
    # === Create save directory structure ===
    level_dir = os.path.join(save_dir, 'Real_Inference')
    deg_dir = os.path.join(level_dir, 'DEG') # Folder for degraded images
    out_dir = os.path.join(level_dir, 'OUT')
    
    os.makedirs(deg_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    total_niqe = 0.0
    total_piqe = 0.0
    count = 0
    
    print(f'\n[' + '='*10 + f' Running Inference on {len(val_loader)} images ' + '='*10 + ']')
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc=f"Inference", ncols=100)):
            fname, img_in = batch
            if isinstance(fname, (list, tuple)):
                fname = fname[0]
                
            if opt.cuda:
                img_in = img_in.cuda()
            
            # Ensure width and height are multiples of 8 to prevent downsampling errors
            _, _, h, w = img_in.shape
            h_new, w_new = h - (h % 8), w - (w % 8)
            img_in = img_in[:, :, :h_new, :w_new]

            # Model inference
            outputs = model(img_in)
            restored_img = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            restored_img = restored_img.clamp(0, 1) # Clamp to 0-1 range
            
            # === Calculate NIQE and PIQE ===
            # Note: pyiqa expects input tensor in [0, 1] range
            current_niqe = iqa_metrics['niqe'](restored_img).item()
            current_piqe = iqa_metrics['piqe'](restored_img).item()
            
            total_niqe += current_niqe
            total_piqe += current_piqe
            count += 1
            
            # === Save images: both aligned original and restored ===
            save_image(img_in, os.path.join(deg_dir, fname))
            save_image(restored_img, os.path.join(out_dir, fname))

    avg_niqe = total_niqe / (count if count > 0 else 1)
    avg_piqe = total_piqe / (count if count > 0 else 1)

    print(f'\n---------- Inference Complete! ----------')
    print(f'Average NIQE: {avg_niqe:.4f} (Lower is better)')
    print(f'Average PIQE: {avg_piqe:.4f} (Lower is better)')
    print(f'Images saved to: {level_dir}')
    
    # Save IQA results to CSV
    csv_path = os.path.join(level_dir, "inference_metrics_report.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'NIQE', 'PIQE'])
        writer.writerow(['Real_Inference', f"{avg_niqe:.4f}", f"{avg_piqe:.4f}"])


# =========================================================================
# Keep evaluate function unchanged (for PSNR/SSIM)
# =========================================================================
def evaluate(model, val_loader, opt, level_name, save_dir):
    model.eval()
    
    # === Create save directory structure ===
    level_dir = os.path.join(save_dir, level_name)
    deg_dir = os.path.join(level_dir, 'DEG')
    tar_dir = os.path.join(level_dir, 'TAR')
    out_dir = os.path.join(level_dir, 'OUT')
    
    os.makedirs(deg_dir, exist_ok=True)
    os.makedirs(tar_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc=f"Testing {level_name}", ncols=100)):
            # Compatible with DataLoader returning filenames
            if len(batch) == 3:
                fname, img_in, img_gt = batch
                if isinstance(fname, (list, tuple)):
                    fname = fname[0]
                if not isinstance(fname, str) or not (fname.endswith('.png') or fname.endswith('.jpg')):
                    fname = f"{i:04d}.png" 
            elif len(batch) == 2:
                img_in, img_gt = batch
                fname = f"{i:04d}.png" 
            else:
                raise ValueError("Dataset returns an unexpected number of elements.")
                
            if opt.cuda:
                img_in, img_gt = img_in.cuda(), img_gt.cuda()
            
            # Ensure width and height are multiples of 8
            _, _, h, w = img_in.shape
            h_new, w_new = h - (h % 8), w - (w % 8)
            img_in = img_in[:, :, :h_new, :w_new]
            img_gt = img_gt[:, :, :h_new, :w_new]

            # Model inference
            outputs = model(img_in)
            restored_img = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            restored_img = restored_img.clamp(0, 1) 
            
            # === Save images ===
            save_image(img_in, os.path.join(deg_dir, fname))
            save_image(img_gt, os.path.join(tar_dir, fname))
            save_image(restored_img, os.path.join(out_dir, fname))
            
            # Prepare numpy format for PSNR / SSIM
            restored_np = restored_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            target_np = img_gt.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            
            current_psnr = psnr(target_np, restored_np, data_range=1.0)
            
            is_multichannel = restored_np.shape[-1] > 1
            current_ssim = ssim(target_np, restored_np, data_range=1.0, 
                                multichannel=is_multichannel, 
                                channel_axis=-1 if is_multichannel else None)
            
            total_psnr += current_psnr
            total_ssim += current_ssim
            count += 1

    avg_psnr = total_psnr / (count if count > 0 else 1)
    avg_ssim = total_ssim / (count if count > 0 else 1)
    
    print(f'\n---------- Result for {level_name}: PSNR = {avg_psnr:.4f} dB, SSIM = {avg_ssim:.4f} ----------')
    return avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser(description="FVMOT Tester")
    
    # === Parameters ===
    parser.add_argument("--weights", type=str, default='./checkpoint/model_sumnot_mri_post_128_bs3.pth', help="Path to the pre-trained model checkpoint (.pt or .pth)")
    parser.add_argument("--backbone", type=str, default="Restormer", help="Model backbone to use")
    parser.add_argument("--understanding_model_name", default="vit_base_patch16_224.mae", type=str, help="To determine projector dim")
    parser.add_argument("--cuda", action='store_true', default=True, help="Use GPU")
    parser.add_argument("--gpus", type=str, default="0", help="Visible GPU ids, e.g. 5 or 2,5")
    parser.add_argument('--data_dir', type=str, default='data/MFMRI', help='Root path of MFMRI dataset')
    parser.add_argument("--save_dir", default="./results/MFMRI/", type=str, help="Directory to save output images and metrics")
    
    # === Added: Pure inference parameter ===
    parser.add_argument("--inference_dir", type=str, default=None, help="Path to raw images folder for pure inference (No GT)")

    # parser.add_argument("--inference_dir", type=str, default='data/m4raw0.3T', help="Path to raw images folder for pure inference (No GT)")
    opt = parser.parse_args()
    if opt.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpus)

    val_levels = [
        'LF_0.055T',
        'LF_0.075T',
        'LF_0.1T',
        'LF_0.3T',
        'LF_0.5T',
        'LF_0.8T',
        'LF_1.0T'
    ]

    # Initialize IQA Metrics globally so they are only loaded into memory once
    device = torch.device('cuda' if opt.cuda else 'cpu')
    if opt.cuda:
        torch.cuda.set_device(0)
    if opt.inference_dir is not None:
        print("===> Initializing NIQE and PIQE metrics...")
        iqa_metrics = {
            'niqe': pyiqa.create_metric('niqe').to(device),
            'piqe': pyiqa.create_metric('piqe').to(device)
        }
    else:
        iqa_metrics = None

    # === 1. Build Network (Strictly align with training code) ===
    # Dynamically determine dimension
    if 'dinov2' in opt.understanding_model_name.lower():
        u_dim = 384
    else:
        u_dim = 768  # MAE default is 768

    print(f"===> Building model {opt.backbone} with projector_out_dim={u_dim}...")
    if opt.backbone == 'Restormer':
        # Added use_projector=True and projector_out_dim=u_dim
        model = Restormer(decoder=True, use_projector=True, projector_out_dim=u_dim)
    else:
        model = Restormer(decoder=True, use_projector=True, projector_out_dim=u_dim)

    if opt.cuda:
        model = model.cuda()

    # === 2. Load Weights ===
    print(f"===> Loading weights from {opt.weights}...")
    checkpoint = torch.load(opt.weights, map_location=device)
    
    # Match your training code checkpoint saving logic: "restoration_net"
    if isinstance(checkpoint, dict):
        if "restoration_net" in checkpoint:  
            target = checkpoint["restoration_net"]
        elif "Restormer" in checkpoint:
            target = checkpoint["Restormer"]
        elif "model" in checkpoint:
            target = checkpoint["model"]
        else:
            target = checkpoint  
    else:
        target = checkpoint  

    # Extract state_dict
    if isinstance(target, torch.nn.Module):
        state_dict = target.state_dict()
    else:
        state_dict = target

    # Clean dictionary, remove 'module.' prefix left by DDP training
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Load parameters 
    model.load_state_dict(new_state_dict, strict=False)
    print("===> Weights loaded successfully!")

    # Ensure root directory exists
    os.makedirs(opt.save_dir, exist_ok=True)
    
    # === 3. Check for pure inference mode ===
    if opt.inference_dir is not None:
        print(f"\n=> Initializing Inference Dataloader for folder: {opt.inference_dir}")
        try:
            real_set = InferenceDataset(opt.inference_dir)
            real_loader = DataLoader(real_set, batch_size=1, shuffle=False)
            if len(real_loader) == 0:
                print(f"[!] Warning: No images found in {opt.inference_dir}")
            else:
                inference_only(model, real_loader, opt, save_dir=opt.save_dir, iqa_metrics=iqa_metrics)
        except Exception as e:
            print(f"[!] Failed to load inference data. Error: {e}")
        return  # Inference complete, exit program directly
    # ==================================

    results = {}
    
    # === 4. Iterate over test sets ===
    for level in val_levels:
        print(f"\n=> Initializing Dataloader for level: {level}")
        try:
            val_set = MFMRI_TestDataset(opt, level=level)
            val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        except Exception as e:
            print(f"[!] Failed to load data for {level}. Error: {e}")
            continue
        
        if len(val_loader) == 0:
            print(f"[!] Warning: Dataloader for {level} is empty, skipping...")
            continue
            
        p, s = evaluate(model, val_loader, opt, level_name=level, save_dir=opt.save_dir)
        results[level] = {'PSNR': p, 'SSIM': s}

    # === 5. Print terminal summary table ===
    print("\n" + "="*50)
    print(" "*13 + "FINAL TESTING REPORT")
    print("="*50)
    print(f"{'Field Level':<15} | {'PSNR (dB)':<12} | {'SSIM':<12}")
    print("-" * 50)
    
    total_p, total_s = 0.0, 0.0
    for level, metrics in results.items():
        print(f"{level:<15} | {metrics['PSNR']:<12.4f} | {metrics['SSIM']:<12.4f}")
        total_p += metrics['PSNR']
        total_s += metrics['SSIM']
        
    print("-" * 50)
    avg_p, avg_s = 0.0, 0.0
    if len(results) > 0:
        avg_p = total_p/len(results)
        avg_s = total_s/len(results)
        print(f"{'AVERAGE':<15} | {avg_p:<12.4f} | {avg_s:<12.4f}")
    print("="*50)

    # === 6. Save results to CSV file ===
    if len(results) > 0:
        csv_path = os.path.join(opt.save_dir, "metrics_report.csv")
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Field Level', 'PSNR (dB)', 'SSIM'])
            for level, metrics in results.items():
                writer.writerow([level, f"{metrics['PSNR']:.4f}", f"{metrics['SSIM']:.4f}"])
            writer.writerow(['AVERAGE', f"{avg_p:.4f}", f"{avg_s:.4f}"])
        print(f"\n[+] Testing metrics successfully saved to: {csv_path}")

if __name__ == "__main__":
    main()