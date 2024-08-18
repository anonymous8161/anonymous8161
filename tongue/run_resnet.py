import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from dataloader import get_dataloaders
from tqdm import tqdm
import os
import hydra
from omegaconf import DictConfig
from torchvision.models import resnet50, ResNet50_Weights
import wandb
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torchvision.ops import sigmoid_focal_loss
from timm.loss import AsymmetricLossMultiLabel
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# PRESET_WEIGHTS = torch.tensor([7.9751, 2.2189, 2.0448, 10.5622, 1.1642, 1.6667, 1.0000, 5.9751]).to(device)
# PRESET_WEIGHTS = torch.tensor([0.2445, 0.0680, 0.0627, 0.3239, 0.0357, 0.0511, 0.0307, 0.1832]).to(device)
# PRESET_WEIGHTS = torch.tensor([8.1900, 2.2779, 2.1013, 10.8438, 1.1955, 1.7123, 1.0273, 6.1314]).to(device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class TongueNet(nn.Module):
    def __init__(self, num_classes):
        super(TongueNet, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        self.sigmoid = nn.Sigmoid() # ASL doesn't need

    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x) # ASL doesn't need
        # return x
    
def calculate_metrics(outputs, labels):
    # outputs = torch.sigmoid(outputs) # ASL
    predicted = (outputs > 0.5).float()
    correct = (predicted == labels)
    return correct, predicted

def calculate_f1_scores(outputs, labels):
    # outputs = torch.sigmoid(outputs) # ASL
    predicted = (outputs > 0.5).cpu().numpy()
    labels = labels.cpu().numpy()
    f1_scores = f1_score(labels, predicted, average=None)
    macro_f1 = f1_score(labels, predicted, average='macro')
    return f1_scores, macro_f1

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, save_dir):
    best_val_acc = 0.0
    num_features = model.resnet.fc.out_features

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)    
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        feature_correct_count = torch.zeros(num_features, dtype=torch.long, device=device)
        all_correct_count = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                correct, predicted = calculate_metrics(outputs, labels)
                feature_correct_count += correct.sum(dim=0)
                all_correct_count += (correct.all(dim=1)).sum().item()
                total_samples += inputs.size(0)
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                all_predictions.append(predicted.cpu())
                all_labels.append(labels.cpu())
        
        val_loss /= len(val_loader.dataset)
        feature_accuracies = feature_correct_count.float() / total_samples
        avg_feature_accuracy = feature_accuracies.mean().item()
        all_correct_accuracy = all_correct_count / total_samples
        
        # Calculate F1 scores
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        f1_scores, macro_f1 = calculate_f1_scores(all_predictions, all_labels)
        avg_f1_score = np.mean(f1_scores)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        # print('Feature Metrics:')
        # for i in range(num_features):
        #     print(f'  Feature {i+1}: Acc={feature_accuracies[i]:.4f}, F1={f1_scores[i]:.4f}')
        print(f'Average Feature Accuracy: {avg_feature_accuracy:.4f}')
        print(f'Average F1 Score: {avg_f1_score:.4f}')
        # print(f'Macro F1 Score: {macro_f1:.4f}')
        print(f'All Correct Accuracy: {all_correct_accuracy:.4f}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        wandb_log_dict = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "val_avg_feature_acc": avg_feature_accuracy,
            "val_avg_f1_score": avg_f1_score,
            # "val_macro_f1_score": macro_f1,
            "val_all_correct_acc": all_correct_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"]
        }
        # for i in range(num_features):
        #     wandb_log_dict[f"val_feature_{i+1}_acc"] = feature_accuracies[i].item()
        #     wandb_log_dict[f"val_feature_{i+1}_f1"] = f1_scores[i]
        
        wandb.log(wandb_log_dict)

        # if avg_feature_accuracy > best_val_acc:
        #     best_val_acc = avg_feature_accuracy
        #     torch.save(model.state_dict(), os.path.join(save_dir, 'resnet_best_model.pth'))
        #     print(f"New best model saved with average validation feature accuracy: {best_val_acc:.4f}")

        if avg_f1_score > best_val_acc:
            best_val_acc = avg_f1_score
            torch.save(model.state_dict(), os.path.join(save_dir, 'resnet_best_model.pth'))
            print(f"New best model saved with average validation f1-score: {best_val_acc:.4f}")
        print(f"Current best f1-score: {best_val_acc:.4f}")
        scheduler.step()

def test_model(model, test_loader):
    model.eval()
    num_features = model.resnet.fc.out_features
    feature_correct_count = torch.zeros(num_features, dtype=torch.long, device=device)
    all_correct_count = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    jaccard_sum = 0.0

    test_pbar = tqdm(test_loader, desc='Testing')
    with torch.no_grad():
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            correct, predicted = calculate_metrics(outputs, labels)
            feature_correct_count += correct.sum(dim=0)
            all_correct_count += (correct.all(dim=1)).sum().item()
            total_samples += inputs.size(0)
            
            all_predictions.append(predicted.cpu())
            all_labels.append(labels.cpu())
            all_probabilities.append(outputs.cpu())

            # Calculate Jaccard Index for this batch
            intersection = torch.logical_and(predicted, labels).sum(dim=1)
            union = torch.logical_or(predicted, labels).sum(dim=1)
            jaccard = intersection.float() / union.float()
            jaccard_sum += jaccard.sum().item()

    feature_accuracies = feature_correct_count.float() / total_samples
    avg_feature_accuracy = feature_accuracies.mean().item()
    all_correct_accuracy = all_correct_count / total_samples

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probabilities = torch.cat(all_probabilities, dim=0)
    f1_scores, macro_f1 = calculate_f1_scores(all_predictions, all_labels)
    avg_f1_score = np.mean(f1_scores)

    # Calculate overall Jaccard Index
    jaccard_index = jaccard_sum / total_samples

    print(f'acc = {feature_accuracies.tolist()}')
    print(f'f1_score = {f1_scores.tolist()}')
    print(f'avg_acc = {avg_feature_accuracy:.4f}')
    print(f'avg_f1 = {avg_f1_score:.4f}')
    print(f'all_correct = {all_correct_accuracy:.4f}')
    print(f'Jaccard Index = {jaccard_index:.4f}')

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
    df.to_csv("roc/resnet_origin_fold5.csv", index=False)

    return avg_feature_accuracy, avg_f1_score, all_correct_accuracy, feature_accuracies.tolist(), f1_scores.tolist(), macro_f1, jaccard_index

@hydra.main(version_base=None, config_path="config", config_name="resnet_origin_test")
def main(cfg):
    print(f"Configuration:\n{cfg}")

    # Set random seed
    set_seed(cfg.seed)
    
    if cfg.mode == "train":
        wandb.init(project="tongue-classification")
        train_loader, val_loader = get_dataloaders(cfg.data.train_csv, cfg.data.val_csv, cfg.data.img_dir,
                                                   cfg.data.batch_size, cfg.data.num_workers)

        model = TongueNet(cfg.model.num_classes).to(device)

        criterion = nn.BCELoss()
        # criterion = nn.BCELoss(weight=PRESET_WEIGHTS)
        # criterion = nn.BCEWithLogitsLoss(weight=PRESET_WEIGHTS)
        # def criterion(outputs, labels):
        #     return sigmoid_focal_loss(outputs, labels, alpha=0.25, gamma=2.0, reduction='mean')
        
        # criterion = AsymmetricLossMultiLabel(
        #                 gamma_neg=4, 
        #                 gamma_pos=0, 
        #                 clip=0.05,
        #                 disable_torch_grad_focal_loss=False
        #             )
          
        optimizer = optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
        
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        os.makedirs(cfg.training.save_dir, exist_ok=True)

        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                    cfg.training.num_epochs, cfg.training.save_dir)

        wandb.finish()

    elif cfg.mode == "test":
        _, test_loader = get_dataloaders(None, cfg.data.val_csv, cfg.data.img_dir,
                                        cfg.data.batch_size, cfg.data.num_workers)

        model = TongueNet(cfg.model.num_classes).to(device)
        model.load_state_dict(torch.load(os.path.join(cfg.training.save_dir, 'resnet_best_model.pth')))

        avg_feature_accuracy, avg_f1_score, all_correct_accuracy, feature_accuracies, feature_f1_scores, macro_f1, jaccard_index = test_model(model, test_loader)
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}. Choose 'train' or 'test'.")

if __name__ == '__main__':
    main()