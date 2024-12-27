import torch
import time
import matplotlib.pyplot as plt
from torch import nn
from dataload import trafficDataset
from torch.utils.data import DataLoader
from resnet import ResNet
from sklearn.metrics import precision_recall_fscore_support
from diagram import plotScore
from torch.amp import autocast, GradScaler
import cProfile
import pstats
import io

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# EPOCH = 50
# LEARNING_RATE = 0.001
EPOCH = 30
LEARNING_RATE = 0.01

full_data = trafficDataset(label_file_path='./Train/label_train.txt', train=True)
train_size = int(0.9 * len(full_data))
vali_size = len(full_data) - train_size
train_data, vali_data = torch.utils.data.random_split(full_data, [train_size, vali_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
vali_loader = DataLoader(vali_data, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
loss_func = nn.CrossEntropyLoss()

def train_faster():
    torch.cuda.empty_cache()
    model = torch.compile(ResNet()).to(device)  # Compile the model for optimized execution
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
    
    scaler = GradScaler('cuda')  # Initialize GradScaler for mixed precision training

    total_step = len(train_loader)

    plot_x = []
    plot_acc = []
    plot_recall = []
    plot_precision = []
    plot_f1 = []
    plot_x_loss = []
    plot_loss = []
    epoch_times = []

    model.train()
    for epoch in range(EPOCH):
        epoch_start = time.time()  # Start time for epoch
        torch.cuda.empty_cache()
        for step, (images, labels) in enumerate(train_loader):
            step_start = time.time()  # Start time for step
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):   # Enable autocast for mixed precision training
                outputs = model(images)
                loss = loss_func(outputs, labels)

            scaler.scale(loss).backward()  # Scale the loss and backpropagate
            scaler.step(optimizer)  # Update the weights
            scaler.update()  # Update the scale for next iteration

            step_end = time.time()  # End time for step
            step_time = step_end - step_start  # Calculate step time

            if (step + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{EPOCH}], Step [{step + 1}/{total_step}], Loss: {loss.item():.4f}, Step Time: {step_time:.4f} seconds')

        scheduler.step()  # Step the learning rate scheduler

        epoch_end = time.time()
        epoch_times.append((epoch_end - epoch_start) / total_step)

        # Validation step can be added here

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

if __name__ == '__main__':
    EPOCH = 1
    pr = cProfile.Profile()
    pr.enable()
    train_faster()
    pr.disable()

    # Print profiling results
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())