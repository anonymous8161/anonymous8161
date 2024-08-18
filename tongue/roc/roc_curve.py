import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def read_csv_files(file_paths):
    return [pd.read_csv(path) for path in file_paths]

def calculate_roc_auc(dfs, num_classes=8):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(num_classes):
        class_tprs = []
        class_aucs = []
        
        for df in dfs:
            y_true = df[f'label_{i}']
            y_score = df[f'prob_{i}']
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            class_tprs.append(np.interp(mean_fpr, fpr, tpr))
            class_tprs[-1][0] = 0.0
            class_aucs.append(auc(fpr, tpr))
        
        mean_tpr = np.mean(class_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        
        tprs.append(mean_tpr)
        aucs.append(mean_auc)

    return mean_fpr, tprs, aucs

def plot_roc_curves(results):
 
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    fig, ax = plt.subplots(figsize=(10, 9))  
    
    colors = sns.color_palette("muted", n_colors=len(results))
    
    for (model, result), color in zip(results.items(), colors):
        mean_tpr = np.mean(result['tprs'], axis=0)
        mean_auc = np.mean(result['aucs'])
        ax.plot(result['mean_fpr'], mean_tpr, color=color, 
                lw=2, label=f'{model} (AUC = {mean_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    ax.set_xlabel('False Positive Rate', fontsize=18, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=18, fontweight='bold')
    ax.set_title('Average ROC Curves', fontsize=18, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=18, frameon=True, fancybox=True, framealpha=0.8, edgecolor='gray')

    legend = ax.legend(loc="lower right", fontsize=18, frameon=True, fancybox=True, 
                       framealpha=0.8, edgecolor='gray', title="Models")
    legend.get_title().set_fontsize('18') 
    legend.get_title().set_fontweight('bold')  
    
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    sns.set_style("whitegrid")
    
    plt.tight_layout(pad=2.0)
    
    plt.show()

file_paths = {
    'ResNet+ATFE': [
        'resnet_align_fold1.csv',
        'resnet_align_fold2.csv',
        'resnet_align_fold3.csv',
        'resnet_align_fold4.csv',
        'resnet_align_fold5.csv'
    ],
    'TransFG': [
        'transfg_fold1.csv',
        'transfg_fold2.csv',
        'transfg_fold3.csv',
        'transfg_fold4.csv',
        'transfg_fold5.csv'
    ],
    'ResNet': [
        'resnet_origin_fold1.csv',
        'resnet_origin_fold2.csv',
        'resnet_origin_fold3.csv',
        'resnet_origin_fold4.csv',
        'resnet_origin_fold5.csv'
    ],
    'SignNet': [ 
        'tongue_fold1.csv',
        'tongue_fold2.csv',
        'tongue_fold3.csv',
        'tongue_fold4.csv',
        'tongue_fold5.csv'
    ],
    'ViT': [
        'vit_fold1.csv',
        'vit_fold2.csv',
        'vit_fold3.csv',
        'vit_fold4.csv',
        'vit_fold5.csv'
    ]

}

results = {}

for model, paths in file_paths.items():
    dfs = read_csv_files(paths)
    mean_fpr, tprs, aucs = calculate_roc_auc(dfs)
    results[model] = {
        'mean_fpr': mean_fpr,
        'tprs': tprs,
        'aucs': aucs
    }

    print(f"\n{model} result:")
    for i, auc_value in enumerate(aucs):
        print(f"Class {i}: AUC = {auc_value:.4f}")
    print(f"Avg AUC: {np.mean(aucs):.4f}")

plot_roc_curves(results)