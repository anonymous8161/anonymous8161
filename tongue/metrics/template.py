import numpy as np


fold1_acc = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
fold1_f1 = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
fold1_avg_acc = 0.00
fold1_avg_f1 = 0.00
fold1_all_correct = 0.00
fold1_jaccard = 0.00  

fold2_acc = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
fold2_f1 = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
fold2_avg_acc = 0.00
fold2_avg_f1 = 0.00
fold2_all_correct = 0.00
fold2_jaccard = 0.00 

fold3_acc = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
fold3_f1 = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
fold3_avg_acc = 0.00
fold3_avg_f1 = 0.00
fold3_all_correct = 0.00
fold3_jaccard = 0.00

fold4_acc = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
fold4_f1 = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
fold4_avg_acc = 0.00
fold4_avg_f1 = 0.00
fold4_all_correct = 0.00
fold4_jaccard = 0.00 

fold5_acc = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
fold5_f1 = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
fold5_avg_acc = 0.00
fold5_avg_f1 = 0.00
fold5_all_correct = 0.00
fold5_jaccard = 0.00  


all_acc = [fold1_acc, fold2_acc, fold3_acc, fold4_acc, fold5_acc]
all_f1 = [fold1_f1, fold2_f1, fold3_f1, fold4_f1, fold5_f1]
all_avg_acc = [fold1_avg_acc, fold2_avg_acc, fold3_avg_acc, fold4_avg_acc, fold5_avg_acc]
all_avg_f1 = [fold1_avg_f1, fold2_avg_f1, fold3_avg_f1, fold4_avg_f1, fold5_avg_f1]
all_correct = [fold1_all_correct, fold2_all_correct, fold3_all_correct, fold4_all_correct, fold5_all_correct]
all_jaccard = [fold1_jaccard, fold2_jaccard, fold3_jaccard, fold4_jaccard, fold5_jaccard]  

acc_array = np.array(all_acc)
f1_array = np.array(all_f1)
avg_acc_array = np.array(all_avg_acc)
avg_f1_array = np.array(all_avg_f1)
all_correct_array = np.array(all_correct)
jaccard_array = np.array(all_jaccard)  

def calculate_stats(data):
    return np.mean(data, axis=0), np.std(data, axis=0, ddof=1)

acc_means, acc_stds = calculate_stats(acc_array)
f1_means, f1_stds = calculate_stats(f1_array)
avg_acc_mean, avg_acc_std = calculate_stats(avg_acc_array)
avg_f1_mean, avg_f1_std = calculate_stats(avg_f1_array)
all_correct_mean, all_correct_std = calculate_stats(all_correct_array)
jaccard_mean, jaccard_std = calculate_stats(jaccard_array)  

for i in range(8):
    print(f"Task {i+1} Accuracy: {acc_means[i]*100:.2f}% ± {acc_stds[i]*100:.2f}%")
print("####################################")

for i in range(8):    
    print(f"Task {i+1} F1 Score: {f1_means[i]*100:.2f}% ± {f1_stds[i]*100:.2f}%")
print("####################################")

print(f"Average Accuracy: {avg_acc_mean*100:.2f}% ± {avg_acc_std*100:.2f}%")
print(f"Average F1 Score: {avg_f1_mean*100:.2f}% ± {avg_f1_std*100:.2f}%")
print(f"All Correct Accuracy: {all_correct_mean*100:.2f}% ± {all_correct_std*100:.2f}%")
print(f"Jaccard Index: {jaccard_mean*100:.2f}% ± {jaccard_std*100:.2f}%")  