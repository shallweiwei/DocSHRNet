import os
import torch
import cv2
import argparse
import logging
import numpy as np
from model.docshrnet import DocSHRNet
from data.dochighlight import DocHighlightDataset
from torch.utils.data import DataLoader
import pyiqa

def load_model(checkpoint_path, device):
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    model = DocSHRNet(img_channel=3, width=32)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint['model_state_dict'] = remove_module_prefix(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def window_process(model, image_tensor, window_size=256, overlap=32, device="cuda"):
    _, c, h, w = image_tensor.shape
    
    if h <= window_size and w <= window_size:
        with torch.no_grad():
            return model(image_tensor)
    
    stride = window_size - overlap
    
    h_blocks = (h - overlap) // stride + (1 if (h - overlap) % stride else 0)
    w_blocks = (w - overlap) // stride + (1 if (w - overlap) % stride else 0)
    
    h_pad = max(0, stride * h_blocks + overlap - h)
    w_pad = max(0, stride * w_blocks + overlap - w)
    
    if h_pad > 0 or w_pad > 0:
        padded = torch.nn.functional.pad(image_tensor, (0, w_pad, 0, h_pad), mode='reflect')
    else:
        padded = image_tensor
        
    output = torch.zeros_like(padded)
    weight_map = torch.zeros((1, 1, padded.shape[2], padded.shape[3]), device=device)
    
    window_patches = []
    window_indices = []
    
    for i in range(h_blocks):
        for j in range(w_blocks):
            h_start = i * stride
            h_end = min(h_start + window_size, padded.shape[2])
            w_start = j * stride
            w_end = min(w_start + window_size, padded.shape[3])
            
            window = padded[:, :, h_start:h_end, w_start:w_end]
            window_patches.append(window)
            window_indices.append((h_start, h_end, w_start, w_end))
    
    for window, (h_start, h_end, w_start, w_end) in zip(window_patches, window_indices):
        with torch.no_grad():
            result = model(window)
        h_size = h_end - h_start
        w_size = w_end - w_start
        window_weight = torch.ones((1, 1, h_size, w_size), device=device)
        output[:, :, h_start:h_end, w_start:w_end] += result * window_weight
        weight_map[:, :, h_start:h_end, w_start:w_end] += window_weight
    
    output = output / (weight_map)
    output = output[:, :, :h, :w]
    
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',default='/data2/xhw/final_500', type=str)
    parser.add_argument('--checkpoint',default='/data2/xhw/dochighlight-main/dochighlightsotadir_ssimcorrect/v6elementadd_ARF_DSA_spatial_0.1perceptual_ssimcorrect/checkpoint_iter_50000.pth', type=str)
    parser.add_argument('--output_dir', type=str, default='./infer_dirsotadochighlight_ssimcorrect/baseline_spatial_0.1perceptual_ssimcorrect')
    parser.add_argument('--eval', default=True, type=bool, help='Whether to perform PSNR/SSIM evaluation')
    parser.add_argument('--window_size', default=1024, type=int, help='Window size for sliding window inference')
    parser.add_argument('--overlap', default=512, type=int, help='Overlap size between adjacent windows')

    args = parser.parse_args()

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.output_dir, 'infer_log')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'window_infer_{timestamp}.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    if args.eval:
        psnr_metric = pyiqa.create_metric('psnr').to(device)
        ssim_metric = pyiqa.create_metric('ssim', test_y_channel=False).to(device)
        psnr_sum, ssim_sum, count = 0.0, 0.0, 0

    infer_dataset = DocHighlightDataset(
        base_dir=args.input_dir,
        split='test',
        spatial_transform=False,
    )
    infer_loader = DataLoader(
        infer_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Starting window inference with window size: {args.window_size}, overlap: {args.overlap}")
    print(f"Starting window inference with window size: {args.window_size}, overlap: {args.overlap}")

    for idx, (degraded_image, gt_image, image_name) in enumerate(infer_loader):
        image_name = image_name[0]
        base_name = os.path.splitext(image_name)[0].replace('_in', '')
        degraded_image = degraded_image.to(device)
        gt_image = gt_image.to(device)
        
        output = window_process(
            model, 
            degraded_image,
            window_size=args.window_size,
            overlap=args.overlap,
            device=device
        )
        
        output = torch.clamp(output, 0, 1)
        output = (output * 255.0).round() / 255.0

        if args.eval:
            psnr_val = psnr_metric(output, gt_image).item()
            ssim_val = ssim_metric(output, gt_image).item()
            psnr_sum += psnr_val
            ssim_sum += ssim_val
            count += 1
            logging.info(f"Image {image_name}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}")
            print(f"Processing image {idx+1}/{len(infer_loader)}: {image_name}, PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}")

        output_np = (output.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        save_path = os.path.join(args.output_dir, f"{base_name}_result.png")
        cv2.imwrite(save_path, cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR))

    if args.eval and count > 0:
        avg_psnr = psnr_sum / count
        avg_ssim = ssim_sum / count
        message = f"Avg PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}"
        print(message)
        logging.info(message)
    elif args.eval:
        print("No images inferred.")
        logging.info("No images inferred.")

if __name__ == '__main__':
    main()