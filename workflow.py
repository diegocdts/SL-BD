import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from train import call_train
from test import call_test
from plot_losses import call_losses

print('Workflow start', datetime.now().strftime("%d/%m/%Y - %H:%M:%S"))

torch.manual_seed(42)

arr = np.loadtxt('/home/src/experiments.csv', delimiter=',', dtype=str, ndmin=2)
loss_paths = []

for line in arr:
    if line[0][0] == '#':
        continue
    epochs = int(line[0])
    lr = float(line[1])
    base_channels = int(line[2])
    train_y_path = line[3]
    test_y_path = line[4]
    train_x_path = line[5]
    test_x_path = line[6]

    dir_name = str(Path(train_y_path).parent.name)

    result_dir = f'/home/src/results/{dir_name}_EP_{epochs}_LR_{lr}_BC_{base_channels}'
    wavelet_title = f'{dir_name}_{base_channels}'
    os.makedirs(result_dir, exist_ok=True)

    mean_y, std_y, mean_x, std_x = call_train(epochs, lr, base_channels, train_y_path, train_x_path, result_dir, wavelet_title)
    loss_paths.append(f'{result_dir}/loss_history.npy')

    call_test(base_channels, result_dir, train_y_path, train_x_path, mean_y, std_y, mean_x, std_x)

    if train_y_path != test_y_path:
        call_test(base_channels, result_dir, test_y_path, test_x_path, mean_y, std_y, mean_x, std_x)
        call_test(base_channels, result_dir, test_y_path, test_x_path, mean_y, std_y, mean_x, std_x, reversed_polarity=True)

call_losses(loss_paths)

print('Workflow end', datetime.now().strftime("%d/%m/%Y - %H:%M:%S"))
