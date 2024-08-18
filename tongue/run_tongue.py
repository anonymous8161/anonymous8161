import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from dataloader3_cam import get_dataloaders
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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import optuna
from network3 import TongueNet
from torch.optim.lr_scheduler import LinearLR, StepLR, SequentialLR
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# class TongueNet(nn.Module):
#     def __init__(self, num_classes):
#         super(TongueNet, self).__init__()
#         self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
#         num_ftrs = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_ftrs, num_classes)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.resnet(x)
#         return self.sigmoid(x)

def calculate_metrics(outputs, labels):
    predicted = (outputs > 0.5).float()
    correct = (predicted == labels)
    return correct, predicted

def calculate_f1_scores(outputs, labels):
    predicted = (outputs > 0.5).cpu().numpy()
    labels = labels.cpu().numpy()
    f1_scores = f1_score(labels, predicted, average=None)
    macro_f1 = f1_score(labels, predicted, average='macro')
    return f1_scores, macro_f1


main_weight = 1.0
color_weight = 0.99658  
fur_weight = 0.62468  
def train_model(model, train_loader, val_loader, criterion_main, criterion_color, criterion_fur, optimizer, scheduler, num_epochs, save_dir, warmup_epochs):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for whole_image, body_image, edge_image, labels, color_labels, fur_label, img_name in train_pbar:
            whole_image, body_image, edge_image = whole_image.to(device), body_image.to(device), edge_image.to(device)
            labels, color_labels, fur_label = labels.to(device), color_labels.to(device), fur_label.to(device)
            optimizer.zero_grad()
            outputs = model(whole_image, body_image, edge_image)

            # Combine all outputs
            attributes_outputs = torch.cat([
                outputs['edge_ensemble_prediction'][:, :2],  # TonguePale, TipSideRed
                outputs['body_ensemble_prediction'][:, 0:1], # RedSpot
                outputs['edge_ensemble_prediction'][:, 2:3], # Ecchymosis
                outputs['crack'],                            # Crack
                outputs['toothmark'],                        # Toothmark
                outputs['body_ensemble_prediction'][:, 1:],  # FurThick, FurYellow
            ], dim=1)
            color_outputs = outputs['color']
            fur_output = outputs['fur']
            # labels: ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack', 'Toothmark', 'FurThick', 'FurYellow']

            # if epoch < warmup_epochs:
            #     color_weight = 5.0
            #     fur_weight = 5.0
            # else:
            #     color_weight = 1.0
            #     fur_weight = 1.0

            loss_main = criterion_main(attributes_outputs, labels)
            loss_color = criterion_color(color_outputs, color_labels)
            loss_fur = criterion_fur(fur_output, fur_label)
            loss = loss_main + color_weight * loss_color + fur_weight * loss_fur
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * whole_image.size(0)
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        feature_correct_count = torch.zeros(8, dtype=torch.long, device=device)
        all_correct_count = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for whole_image, body_image, edge_image, labels, color_labels, fur_label, img_name in val_pbar:
                whole_image, body_image, edge_image = whole_image.to(device), body_image.to(device), edge_image.to(device)
                labels, color_labels, fur_label = labels.to(device), color_labels.to(device), fur_label.to(device)
                
                outputs = model(whole_image, body_image, edge_image)
                attributes_outputs = torch.cat([
                    outputs['edge_ensemble_prediction'][:, :2],   # TonguePale, TipSideRed
                    outputs['body_ensemble_prediction'][:, 0:1],  # RedSpot
                    outputs['edge_ensemble_prediction'][:, 2:3],  # Ecchymosis
                    outputs['crack'],                             # Crack
                    outputs['toothmark'],                         # Toothmark
                    outputs['body_ensemble_prediction'][:, 1:],   # FurThick, FurYellow
                ], dim=1)
                color_outputs = outputs['color']
                fur_output = outputs['fur']

                loss_main = criterion_main(attributes_outputs, labels)
                loss_color = criterion_color(color_outputs, color_labels)
                loss_fur = criterion_fur(fur_output, fur_label)
                loss = loss_main + loss_color + loss_fur

                val_loss += loss.item() * whole_image.size(0)
                
                correct, predicted = calculate_metrics(attributes_outputs, labels)
                feature_correct_count += correct.sum(dim=0)
                all_correct_count += (correct.all(dim=1)).sum().item()
                total_samples += whole_image.size(0)
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                all_predictions.append(predicted.cpu())
                all_labels.append(labels.cpu())
        
        val_loss /= len(val_loader.dataset)
        feature_accuracies = feature_correct_count.float() / total_samples
        avg_feature_accuracy = feature_accuracies.mean().item()
        all_correct_accuracy = all_correct_count / total_samples
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        f1_scores, macro_f1 = calculate_f1_scores(all_predictions, all_labels)
        avg_f1_score = np.mean(f1_scores)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Average Feature Accuracy: {avg_feature_accuracy:.4f}')
        print(f'Average F1 Score: {avg_f1_score:.4f}')
        print(f'All Correct Accuracy: {all_correct_accuracy:.4f}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        wandb_log_dict = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "val_avg_feature_acc": avg_feature_accuracy,
            "val_avg_f1_score": avg_f1_score,
            "val_all_correct_acc": all_correct_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"]
        }
        
        wandb.log(wandb_log_dict)

        if avg_f1_score > best_val_acc:
            best_val_acc = avg_f1_score
            torch.save(model.state_dict(), os.path.join(save_dir, 'resnet_best_model.pth'))
            print(f"New best model saved with average validation f1-score: {best_val_acc:.4f}")
        print(f"Current best f1-score: {best_val_acc:.4f}")
        scheduler.step()

def test_model(model, test_loader):
    model.eval()
    feature_correct_count = torch.zeros(8, dtype=torch.long, device=device)
    all_correct_count = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    jaccard_sum = 0.0

    output_dir = 'tonuge_cam'
    os.makedirs(output_dir, exist_ok=True)

    # target_layers = [model.resnet.layer4[-1]]
    # cam = GradCAM(model=model, target_layers=target_layers)
    class_names = ['TonguePale', 'TipSideRed', 'RedSpot', 'Ecchymosis', 'Crack', 'Toothmark', 'FurThick', 'FurYellow']

    test_pbar = tqdm(test_loader, desc='Testing')
    for whole_image, body_image, edge_image, labels, color_labels, fur_label, img_name in test_pbar:
        whole_image, body_image, edge_image = whole_image.to(device), body_image.to(device), edge_image.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(whole_image, body_image, edge_image)
            attributes_outputs = torch.cat([
                outputs['edge_ensemble_prediction'][:, :2],   # TonguePale, TipSideRed
                outputs['body_ensemble_prediction'][:, 0:1],  # RedSpot
                outputs['edge_ensemble_prediction'][:, 2:3],  # Ecchymosis
                outputs['crack'],                             # Crack
                outputs['toothmark'],                         # Toothmark
                outputs['body_ensemble_prediction'][:, 1:],   # FurThick, FurYellow
            ], dim=1)

            correct, predicted = calculate_metrics(attributes_outputs, labels)
            feature_correct_count += correct.sum(dim=0)
            all_correct_count += (correct.all(dim=1)).sum().item()
            total_samples += whole_image.size(0)
            
            all_predictions.append(predicted.cpu())
            all_labels.append(labels.cpu())
            all_probabilities.append(attributes_outputs.cpu())

            intersection = torch.logical_and(predicted, labels).sum(dim=1)
            union = torch.logical_or(predicted, labels).sum(dim=1)
            jaccard = intersection.float() / union.float()
            jaccard_sum += jaccard.sum().item()

        # for i, class_name in enumerate(class_names):
        #     target_category = ClassifierOutputTarget(i)

        #     grayscale_cam = cam(input_tensor=whole_image, targets=[target_category])
            
        #     input_np = whole_image[0].cpu().numpy().transpose(1, 2, 0)
        #     mean = np.array([0.485, 0.456, 0.406])
        #     std = np.array([0.229, 0.224, 0.225])
        #     input_np = std * input_np + mean
        #     input_np = np.clip(input_np, 0, 1)

        #     cam_image = show_cam_on_image(input_np, grayscale_cam[0, :], use_rgb=True)

        #     img_name = img_name[0].replace(".jpg", "").replace("test/", "")
        #     cam_image_name = f"{img_name}_{class_name}.jpg"
        #     cam_image_path = os.path.join(output_dir, cam_image_name)
        #     cv2.imwrite(cam_image_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    feature_accuracies = feature_correct_count.float() / total_samples
    avg_feature_accuracy = feature_accuracies.mean().item()
    all_correct_accuracy = all_correct_count / total_samples

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probabilities = torch.cat(all_probabilities, dim=0)
    f1_scores, macro_f1 = calculate_f1_scores(all_predictions, all_labels)
    avg_f1_score = np.mean(f1_scores)

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
    df.to_csv("roc/tongue_fold5.csv", index=False)

    return avg_feature_accuracy, avg_f1_score, all_correct_accuracy, feature_accuracies.tolist(), f1_scores.tolist(), macro_f1, jaccard_index

@hydra.main(version_base=None, config_path="config", config_name="tongue_test")
def main(cfg):
    print(f"Configuration:\n{cfg}")

    set_seed(cfg.seed)
    
    if cfg.mode == "train":
        wandb.init(project="tongue-classification")
        train_loader, val_loader = get_dataloaders(cfg.data.train_csv, cfg.data.val_csv, cfg.data.whole_img_dir,
                                                   cfg.data.body_img_dir, cfg.data.edge_img_dir,
                                                   cfg.data.batch_size, cfg.data.num_workers)

        model = TongueNet(cfg.model.num_classes, num_attn_layers=1, hidden_dim=896).to(device)

        main_weight = torch.tensor([3.7461, 1.0419, 0.9613, 4.9609, 0.5468, 0.7833, 0.4699, 2.8043]).to(device)
        color_weight = torch.tensor([0.3283, 1.9589, 3.4653, 0.6714]).to(device)
        criterion_main = nn.BCELoss(weight=main_weight) #not pos_weight
        criterion_color = nn.BCELoss(weight=color_weight)
        criterion_fur = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        warmup_epochs = 3
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)


        step_scheduler = StepLR(optimizer, step_size=16, gamma=0.1)


        scheduler = SequentialLR(optimizer, 
                                 schedulers=[warmup_scheduler, step_scheduler],
                                 milestones=[warmup_epochs])

        os.makedirs(cfg.training.save_dir, exist_ok=True)

        train_model(model, train_loader, val_loader, criterion_main, criterion_color, criterion_fur, optimizer, scheduler,
                    cfg.training.num_epochs, cfg.training.save_dir, warmup_epochs)

        wandb.finish()

    elif cfg.mode == "test":
        _, test_loader = get_dataloaders(None, cfg.data.val_csv, cfg.data.whole_img_dir,
                                         cfg.data.body_img_dir, cfg.data.edge_img_dir,
                                         cfg.data.batch_size, cfg.data.num_workers)

        model = TongueNet(cfg.model.num_classes, num_attn_layers=1, hidden_dim=896).to(device)
        model.load_state_dict(torch.load(os.path.join(cfg.training.save_dir, 'resnet_best_model.pth')))

        avg_feature_accuracy, avg_f1_score, all_correct_accuracy, feature_accuracies, feature_f1_scores, macro_f1, jaccard_index = test_model(model, test_loader)
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}. Choose 'train' or 'test'.")

if __name__ == '__main__':
    main()