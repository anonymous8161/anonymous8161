# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import time

from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataloader import get_dataloaders  

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
import wandb
from sklearn.metrics import accuracy_score, f1_score
import hydra

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def save_model(cfg, model, optimizer, epoch, best_acc):
    model_to_save = model.module if hasattr(model, 'module') else model
    save_path = os.path.join(cfg.output_dir, f"{cfg.name}_best.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, save_path)
    logger.info(f"Best model saved to {save_path}")

def setup(cfg): 
    config = CONFIGS[cfg.model_type]

    num_classes = 8
    model = VisionTransformer(config, cfg.img_size, zero_head=True, num_classes=num_classes, smoothing_value=cfg.smoothing_value)

    model.load_from(np.load(cfg.pretrained_dir))
    if cfg.pretrained_model is not None:
        pretrained_model = torch.load(cfg.pretrained_model)['model']
        model.load_state_dict(pretrained_model)
    device = torch.device('cuda:0')
    print("start to load model to GPU")
    model.to(device)
    num_params = count_parameters(model)
    print("model loaded")
    logger.info(f"{config}")
    logger.info(f"Training parameters {cfg}")
    logger.info(f"Total Parameter: \t{num_params:.1f}M")
    return config, model

def valid(cfg, model, writer, test_loader, global_step):  
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info(f"  Num steps = {len(test_loader)}")
    logger.info(f"  Batch size = {cfg.eval_batch_size}")

    model.eval()
    all_preds = []
    all_labels = []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    loss_fct = nn.BCEWithLogitsLoss()  
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(cfg.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)
            eval_loss = loss_fct(logits, y.float())  
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

            preds = (torch.sigmoid(logits) > 0.5).float()  

        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
        epoch_iterator.set_description(f"Validating... (loss={eval_losses.val:.5f})")

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    accuracy = accuracy_score(all_labels, all_preds)  
    f1 = f1_score(all_labels, all_preds, average='macro')  

    logger.info("\n")
    logger.info("Validation Results")
    logger.info(f"Global Steps: {global_step}")
    logger.info(f"Valid Loss: {eval_losses.avg:.5f}")
    logger.info(f"Valid Accuracy: {accuracy:.5f}")  
    logger.info(f"Valid F1 Score: {f1:.5f}") 
    
    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step) 
    writer.add_scalar("test/f1_score", scalar_value=f1, global_step=global_step) 
        
    return accuracy

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def valid(cfg, model, writer, test_loader, epoch):
    device = torch.device('cuda:0')
    eval_losses = AverageMeter()
    num_features = 8  

    logger.info("***** Running Validation *****")
    logger.info(f"  Num steps = {len(test_loader)}")
    logger.info(f"  Batch size = {cfg.eval_batch_size}")

    model.eval()
    feature_correct_count = torch.zeros(num_features, dtype=torch.long, device=cfg.device)
    all_correct_count = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    loss_fct = nn.BCEWithLogitsLoss()
    val_pbar = tqdm(test_loader, desc=f'Validating Epoch {epoch}')
    with torch.no_grad():
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device) 
            logits = model(inputs)
            logits = model(inputs)
            loss = loss_fct(logits, labels.float())
            eval_losses.update(loss.item())

            outputs = torch.sigmoid(logits)
            correct, predicted = calculate_metrics(outputs, labels)
            feature_correct_count += correct.sum(dim=0)
            all_correct_count += (correct.all(dim=1)).sum().item()
            total_samples += inputs.size(0)
            
            all_predictions.append(predicted.cpu())
            all_labels.append(labels.cpu())
            
            val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    feature_accuracies = feature_correct_count.float() / total_samples
    avg_feature_accuracy = feature_accuracies.mean().item()
    all_correct_accuracy = all_correct_count / total_samples

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    f1_scores, macro_f1 = calculate_f1_scores(all_predictions, all_labels)
    avg_f1_score = np.mean(f1_scores)

    print("\n")
    print(f"Validation Results for Epoch {epoch}")
    print(f"Valid Loss: {eval_losses.avg:.4f}")
    print("Feature Metrics:")
    for i in range(num_features):
        print(f"  Feature {i+1}: Acc={feature_accuracies[i]:.4f}, F1={f1_scores[i]:.4f}")
    print(f"Average Feature Accuracy: {avg_feature_accuracy:.4f}")
    print(f"Average F1 Score: {avg_f1_score:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"All Correct Accuracy: {all_correct_accuracy:.4f}")

    writer.add_scalar("test/loss", scalar_value=eval_losses.avg, global_step=epoch)
    writer.add_scalar("test/avg_feature_accuracy", scalar_value=avg_feature_accuracy, global_step=epoch)
    writer.add_scalar("test/avg_f1_score", scalar_value=avg_f1_score, global_step=epoch)
    writer.add_scalar("test/macro_f1", scalar_value=macro_f1, global_step=epoch)
    writer.add_scalar("test/all_correct_accuracy", scalar_value=all_correct_accuracy, global_step=epoch)

    for i in range(num_features):
        writer.add_scalar(f"test/feature_{i+1}_accuracy", scalar_value=feature_accuracies[i], global_step=epoch)
        writer.add_scalar(f"test/feature_{i+1}_f1", scalar_value=f1_scores[i], global_step=epoch)

    return avg_feature_accuracy

def calculate_metrics(outputs, labels):
    predicted = (outputs > 0.5).float()
    correct = (predicted == labels)
    return correct, predicted

def calculate_f1_scores(outputs, labels):
    predicted = outputs.numpy()
    labels = labels.numpy()
    f1_scores = f1_score(labels, predicted, average=None)
    macro_f1 = f1_score(labels, predicted, average='macro')
    return f1_scores, macro_f1
loss_fct = nn.BCEWithLogitsLoss()

def train(cfg, model):
    device = torch.device('cuda:0')
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    writer = SummaryWriter(log_dir=os.path.join("logs", cfg.name))
    print("start to load image data")
    train_loader, test_loader = get_dataloaders(cfg.data.train_csv, cfg.data.val_csv, cfg.data.img_dir,
                                                   cfg.data.batch_size, cfg.data.num_workers)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.learning_rate,
                                momentum=0.9,
                                weight_decay=cfg.weight_decay)
    scheduler = StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {cfg.num_epochs}")
    logger.info(f"  Instantaneous batch size per GPU = {cfg.train_batch_size}")

    model.zero_grad()
    set_seed(cfg)
    losses = AverageMeter()
    best_acc = 0
    best_model_path = None
    start_time = time.time()

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc=f"Epoch {epoch+1}/{cfg.num_epochs} [Train]",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for inputs, labels in epoch_iterator:
            inputs, labels = inputs.to(device), labels.to(device) 
            logits = model(inputs)
            loss = loss_fct(logits, labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            losses.update(loss.item())
            optimizer.step()
            optimizer.zero_grad()

            epoch_iterator.set_postfix({'loss': f'{losses.val:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})

        accuracy = valid(cfg, model, writer, test_loader, epoch)
        if best_acc < accuracy:
            if best_model_path:
                os.remove(best_model_path) 
            best_model_path = save_model(cfg, model, optimizer, epoch, accuracy)
            best_acc = accuracy
        
        writer.add_scalar("train/loss", scalar_value=losses.avg, global_step=epoch)
        writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=epoch)
        
        scheduler.step()

    logger.info(f"Best Accuracy: \t{best_acc:.6f}")
    logger.info("End Training!")
    end_time = time.time()
    logger.info(f"Total Training Time: \t{(end_time - start_time) / 3600:.6f}")

    writer.close()

def set_seed(cfg): 
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.n_gpu > 0:
        torch.cuda.manual_seed_all(cfg.seed)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.info(f"Device: {device}, n_gpu: {cfg.n_gpu}")

    # Set seed
    set_seed(cfg)

    # Model Setup
    config, model = setup(cfg)

    # Training
    train(cfg, model)


if __name__ == "__main__":
    main()
