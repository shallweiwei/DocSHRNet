import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import argparse

from model.docshrnet import DocSHRNet 
import time
import math  
import logging  
from pathlib import Path  

from data.dochighlight import DocHighlightDataset

from datetime import datetime 
import pyiqa 
import random 
import numpy as np  
from torchvision import models  

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MultiLayerPerceptualLoss(nn.Module):
    def __init__(self, layers=[5, 10, 19]):
        super(MultiLayerPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layers = layers
        self.vgg_blocks = nn.ModuleList()
        start = 0
        for l in layers:
            block = nn.Sequential(*list(vgg.children())[start:l])
            for param in block.parameters():
                param.requires_grad = False
            self.vgg_blocks.append(block)
            start = l

    def forward(self, x, y):
        loss = 0
        for block in self.vgg_blocks:
            fx = block(x)
            fy = block(y)
            loss += nn.functional.l1_loss(fx, fy)
            x, y = fx, fy
        return loss

class StyleLoss(nn.Module):
    def __init__(self, layers=[5, 10, 19]):
        super(StyleLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.vgg_blocks = nn.ModuleList()
        start = 0
        for l in layers:
            block = nn.Sequential(*list(vgg.children())[start:l])
            for param in block.parameters():
                param.requires_grad = False
            self.vgg_blocks.append(block)
            start = l

    def forward(self, x, y):
        style_loss = 0
        for block in self.vgg_blocks:
            fx = block(x)
            fy = block(y)
            gram_fx = self.gram_matrix(fx)
            gram_fy = self.gram_matrix(fy)
            style_loss += nn.functional.l1_loss(gram_fx, gram_fy)
            x, y = fx, fy
        return style_loss

    def gram_matrix(self, feat):
        b, c, h, w = feat.size()
        feat_reshaped = feat.view(b, c, h*w)
        gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))
        return gram / (c * h * w)

def main():
    parser = argparse.ArgumentParser(description='Train a model for document image restoration.')
    parser.add_argument('--base_dir', default='/data2/xhw/final_500', type=str, help='Base directory of images')
    parser.add_argument('--batch_size', default=4, type=int, help='Mini-batch size')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='Initial learning rate')
    parser.add_argument('--output_dir', default='dochighlightsotadir_ssimcorrect/elementaddv6_dsa_spatial_0.1perceptual_ssimcorrect', type=str, help='Directory to save outputs and logs')
    parser.add_argument('--resume', default='', type=str, help='Path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default='', type=str, help='Path to pretrained model weights (default: none)')
    parser.add_argument('--total_iters', default=50000, type=int, help='Total number of iterations to run')
    parser.add_argument('--print_loss_freq', default=50, type=int, help='Frequency of printing training loss (in iterations)')
    parser.add_argument('--test_model_freq', default=10000, type=int, help='Frequency of testing the model (in iterations)')
    parser.add_argument('--save_freq', default=10000, type=int, help='Frequency of saving model checkpoints (in iterations)')
    args = parser.parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    dist.init_process_group(backend="nccl", init_method='env://')
    seed_everything(42)

    if local_rank == 0:
        if not os.path.isdir(args.base_dir):
            raise ValueError(f"Base directory does not exist: {args.base_dir}")
        args.save_path = args.output_dir
        args.log_dir = os.path.join(args.output_dir, 'log')
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = os.path.join(args.log_dir, f'{timestamp}.log')
        
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(log_file_path),
                                logging.StreamHandler()
                            ])
        
        logging.info(vars(args))
        logging.info("Training started")

    train_dataset = DocHighlightDataset(
        base_dir=args.base_dir,
        split='train',
        spatial_transform=True,
    )
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    val_dataset = DocHighlightDataset(
        base_dir=args.base_dir,  # 只用base_dir
        split='test',
        spatial_transform=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = DocSHRNet(img_channel=3, width=32)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    criterion = nn.L1Loss()
    perceptual_criterion = MultiLayerPerceptualLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iters, eta_min=1e-5)

    start_iter = 0
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            logging.info(f"=> loading pretrained model '{args.pretrained}'")
            checkpoint = torch.load(args.pretrained, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"=> loaded pretrained model '{args.pretrained}'")
        else:
            logging.info(f"=> no pretrained model found at '{args.pretrained}'")
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_iter = checkpoint['iter']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info(f"=> loaded checkpoint '{args.resume}' (iter {checkpoint['iter']})")
        else:
            logging.info(f"=> no checkpoint found at '{args.resume}'")

    iteration_losses = []
    scaler = torch.amp.GradScaler()
    train_dataset_len = len(train_dataset)
    val_dataset_len = len(val_dataset)
    if train_dataset_len > 0:
        total_epochs = math.ceil(args.total_iters / len(train_loader))
    else:
        if local_rank == 0:
            logging.info("Training data is empty, cannot train.")
        return
    if local_rank == 0:
        logging.info(f'train_dataset_len: {train_dataset_len}')
        logging.info(f'val_dataset_len: {val_dataset_len}')
        logging.info(f"Total epochs: {total_epochs}")
    
    initial_iter = start_iter
    train_elapsed_time = 0.0
    start_epoch = start_iter // len(train_loader) if len(train_loader) > 0 else 0

    psnr_metric = pyiqa.create_metric('psnr').to(device)
    ssim_metric = pyiqa.create_metric('ssim', test_y_channel=False).to(device)
    ssim_loss_fn = pyiqa.create_metric('ssim',test_y_channel=False, as_loss=True).to(device)

    model.train()
    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        data_iter = iter(train_loader)
        for batch_idx in range(len(train_loader)):
            if start_iter >= args.total_iters:
                break
            iteration_start_time = time.time()
            degraded_patch, gt_patch = next(data_iter)
            if degraded_patch is None or gt_patch is None:
                logging.warning(f"Skipped iteration {start_iter} due to data loading issue.")
                continue
            degraded_patch = degraded_patch.to(device, non_blocking=True)
            gt_patch = gt_patch.to(device, non_blocking=True)
            optimizer.zero_grad()

            outputs = model(degraded_patch)
            l1_loss = criterion(outputs, gt_patch)
            perceptual_loss = 0.1 * perceptual_criterion(outputs, gt_patch)
            ssim_loss = 1 - ssim_loss_fn(outputs, gt_patch)
            total_loss = l1_loss+ perceptual_loss + ssim_loss  
            total_loss.backward()
            optimizer.step()
            iteration_losses.append(total_loss.item())

            iteration_end_time = time.time()
            train_elapsed_time += (iteration_end_time - iteration_start_time)

            if (start_iter + 1) % args.print_loss_freq == 0 and local_rank == 0:
                elapsed_time = train_elapsed_time
                completed_iters = start_iter - initial_iter
                if completed_iters > 0:
                    time_per_iter = elapsed_time / completed_iters
                    remaining_iters = args.total_iters - start_iter - 1
                    remaining_time = time_per_iter * remaining_iters
                else:
                    time_per_iter = 0
                    remaining_time = 0
                current_lr = optimizer.param_groups[0]['lr']
                log_message = ( f'Iter [{start_iter+1}/{args.total_iters}], L1 Loss: {l1_loss.item():.4f}, '
                                f'Perceptual Loss: {perceptual_loss.item():.4f}, '
                                f'SSIM Loss: {ssim_loss.item():.4f}, '
                                f'Total Loss: {total_loss.item():.4f}, '
                                f'Remaining Time: {remaining_time/3600:.2f}h, Learning Rate: {current_lr:.6f}')
                logging.info(log_message)
            
            scheduler.step()

            if (start_iter + 1) % args.test_model_freq == 0:
                val_model(model, val_loader, device, start_iter, args.total_iters, local_rank, psnr_metric, ssim_metric)

            if (start_iter + 1) % args.save_freq == 0 and local_rank == 0:
                save_checkpoint(model, optimizer, scheduler, start_iter, args.save_path)
            
            start_iter += 1

    if local_rank == 0:
        logging.info("Training finished")
    dist.destroy_process_group()

def val_model(model, val_loader, device, start_iter, total_iters, local_rank, psnr_metric, ssim_metric):
    model.eval()
    psnr_total = 0
    ssim_total = 0
    with torch.no_grad():
        for degraded_image, gt_image, image_name in val_loader:
            degraded_image = degraded_image.to(device, non_blocking=True)
            gt_image = gt_image.to(device, non_blocking=True)
            outputs = model(degraded_image)
            outputs = torch.clamp(outputs, 0.0, 1.0)
            output = (outputs * 255.0).round() / 255.0
            psnr = psnr_metric(output, gt_image)
            ssim = ssim_metric(output, gt_image)
            psnr_total += psnr.item()
            ssim_total += ssim.item()
    avg_psnr = psnr_total / len(val_loader)
    avg_ssim = ssim_total / len(val_loader)
    if local_rank == 0:
        val_log_message = f'Iter [{start_iter+1}/{total_iters}], Val PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}'
        logging.info(val_log_message)
    model.train()

def save_checkpoint(model, optimizer, scheduler, start_iter, save_path):
    checkpoint = {
        'iter': start_iter+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    save_path = os.path.join(save_path, f'checkpoint_iter_{start_iter+1}.pth')
    torch.save(checkpoint, save_path)
    logging.info(f'Checkpoint saved at iteration {start_iter+1}')

if __name__ == "__main__":
    main()