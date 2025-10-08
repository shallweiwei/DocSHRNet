import os
import cv2
import torch
import argparse
import pyiqa
import logging 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str,default='', help='Directory of predicted images')
    parser.add_argument('--gt_dir', type=str, default='', help='Directory of ground-truth images')
    args = parser.parse_args()

    log_dir = os.path.join(args.pred_dir, 'eval_log') 
    os.makedirs(log_dir, exist_ok=True)            
    from datetime import datetime                    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(log_dir, f'eval_{timestamp}.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr_metric = pyiqa.create_metric('psnr').to(device)
    ssim_metric = pyiqa.create_metric('ssim',test_y_channel=False).to(device)

    pred_images = sorted([f for f in os.listdir(args.pred_dir) if f.endswith(('result.png'))])
    gt_images = sorted([f for f in os.listdir(args.gt_dir) if f.endswith(('gt.jpg'))])
    
    psnr_total, ssim_total, count = 0.0, 0.0, 0
    for pred_name, gt_name in zip(pred_images, gt_images):
        pred_path = os.path.join(args.pred_dir, pred_name)
        gt_path = os.path.join(args.gt_dir, gt_name)

        pred_image = cv2.imread(pred_path)
        pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
        pred_tensor = torch.from_numpy(pred_image.transpose(2, 0, 1)).float() / 255.0
        pred_tensor = pred_tensor.unsqueeze(0).to(device)

        gt_image = cv2.imread(gt_path)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        gt_tensor = torch.from_numpy(gt_image.transpose(2, 0, 1)).float() / 255.0
        gt_tensor = gt_tensor.unsqueeze(0).to(device)

        pred_tensor = torch.clamp(pred_tensor, 0, 1)
        psnr_val = psnr_metric(pred_tensor, gt_tensor).item()
        ssim_val = ssim_metric(pred_tensor, gt_tensor).item()
        psnr_total += psnr_val
        ssim_total += ssim_val
        count += 1
        logging.info(f"Image: {pred_name} - PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}")

    if count > 0:
        message = f"Average PSNR: {psnr_total / count:.4f}, SSIM: {ssim_total / count:.4f}"
        print(message)
        logging.info(message)
    else:
        print("No valid image pairs found!")
        logging.info("No valid image pairs found!")

if __name__ == '__main__':
    main()