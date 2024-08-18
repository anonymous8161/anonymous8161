# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from dataloader import get_dataloaders
from models.modeling import VisionTransformer, CONFIGS
from sklearn.metrics import accuracy_score, f1_score
import hydra
from omegaconf import DictConfig
import pandas as pd

logger = logging.getLogger(__name__)

def setup(cfg):
    config = CONFIGS[cfg.model_type]
    num_classes = 8
    model = VisionTransformer(config, cfg.img_size, zero_head=True, num_classes=num_classes, smoothing_value=cfg.smoothing_value)
    return config, model

def load_model(cfg, model):
    checkpoint = torch.load(cfg.best_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from {cfg.best_model_path}")
    return model

def test(cfg, model, test_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    num_features = 8 
    feature_correct_count = torch.zeros(num_features, dtype=torch.float, device=device) 
    all_correct_count = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    jaccard_sum = 0.0
    test_loss = 0
    loss_fct = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            
            loss = loss_fct(logits, labels.float())
            test_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            correct = (preds == labels).float()
            feature_correct_count += correct.sum(dim=0)
            all_correct_count += (correct.all(dim=1)).sum().item()
            total_samples += inputs.size(0)
            
            all_predictions.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probabilities.append(probs.cpu())

            # Calculate Jaccard Index for this batch
            intersection = torch.logical_and(preds, labels).sum(dim=1)
            union = torch.logical_or(preds, labels).sum(dim=1)
            jaccard = intersection.float() / union.float()
            jaccard_sum += jaccard.sum().item()

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probabilities = torch.cat(all_probabilities, dim=0)

    feature_accuracies = feature_correct_count / total_samples
    avg_feature_accuracy = feature_accuracies.mean().item()
    all_correct_accuracy = all_correct_count / total_samples

    f1_scores = f1_score(all_labels.numpy(), all_predictions.numpy(), average=None)
    macro_f1 = f1_score(all_labels.numpy(), all_predictions.numpy(), average='macro')
    avg_f1_score = np.mean(f1_scores)

    jaccard_index = jaccard_sum / total_samples
    test_loss /= len(test_loader)

    print(f'acc = {feature_accuracies.tolist()}')
    print(f'f1_score = {f1_scores.tolist()}')
    print(f'avg_acc = {avg_feature_accuracy:.4f}')
    print(f'avg_f1 = {avg_f1_score:.4f}')
    print(f'all_correct = {all_correct_accuracy:.4f}')
    print(f'Jaccard Index = {jaccard_index:.4f}')
    print(f'Test Loss = {test_loss:.4f}')
    print(f'Macro F1 Score = {macro_f1:.4f}')

    # Prepare data for CSV
    results = []
    for labels, probs in zip(all_labels.numpy(), all_probabilities.numpy()):
        row = np.concatenate([labels, probs])
        results.append(row)

    # Create column names
    num_classes = all_labels.shape[1]
    columns = [f'label_{i}' for i in range(num_classes)] + [f'prob_{i}' for i in range(num_classes)]

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results, columns=columns)
    df.to_csv('transfg_fold5.csv', index=False)

    return {
        'feature_accuracies': feature_accuracies.tolist(),
        'f1_scores': f1_scores.tolist(),
        'avg_feature_accuracy': avg_feature_accuracy,
        'avg_f1_score': avg_f1_score,
        'all_correct_accuracy': all_correct_accuracy,
        'jaccard_index': jaccard_index,
        'test_loss': test_loss,
        'macro_f1': macro_f1
    }

@hydra.main(version_base=None, config_path="config", config_name="test_config")
def main(cfg: DictConfig):
 
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    _, test_loader = get_dataloaders(cfg.data.train_csv, cfg.data.test_csv, cfg.data.img_dir,
                                     cfg.data.batch_size, cfg.data.num_workers)

    _, model = setup(cfg)

    model = load_model(cfg, model)

    test(cfg, model, test_loader)

if __name__ == "__main__":
    main()