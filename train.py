import torch
import time
import matplotlib.pyplot as plt
from torch import nn
from dataload import trafficDataset
from torch.utils.data import DataLoader
from resnet import ResNet
from sklearn.metrics import precision_recall_fscore_support
from diagram import plotScore

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# EPOCH = 50
# LEARNING_RATE = 0.001
EPOCH = 30
LEARNING_RATE = 0.01

full_data = trafficDataset(label_file_path='./Train/label_train.txt', train=True)
train_size = int(0.9 * len(full_data))
vali_size = len(full_data) - train_size
train_data, vali_data = torch.utils.data.random_split(full_data, [train_size, vali_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
vali_loader = DataLoader(vali_data, batch_size=32, shuffle=True)
loss_func = nn.CrossEntropyLoss()


def train():
    torch.cuda.empty_cache()
    model = ResNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    total_step = len(train_loader)

    plot_x = []
    plot_acc = []
    plot_recall = []
    plot_precision = []
    plot_f1 = []
    plot_x_loss = []
    plot_loss = []
    epoch_times = []

    for epoch in range(EPOCH):
        epoch_start = time.time()  # Start time for epoch
        torch.cuda.empty_cache()
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                print(f'Epoch [{epoch+1}/{EPOCH}], Step [{step+1}/{total_step}], Loss: {loss.item():.7f}')
            if step % 50 == 0:
                plot_x_loss.append(step / total_step + epoch)
                plot_loss.append(loss.item())
        scheduler.step()

        epoch_time = time.time() - epoch_start  # Calculate epoch duration
        epoch_times.append(epoch_time / total_step)

        # Validation and metrics calculation
        true_target = torch.FloatTensor()
        pre_target = torch.FloatTensor()
        correct_vali = 0
        total_vali = 0
        for images_vali, target in vali_loader:
            images_vali = images_vali.to(device)
            target = target.to(device)
            with torch.no_grad():
                pred = model(images_vali)
                maxk = max((1, 5))
                target_resize = target.view(-1, 1)
                _, predicted = pred.topk(maxk, 1, True, True)
                correct_vali += torch.eq(predicted, target_resize).sum().item()
                total_vali += target.size(0)

                temp = predicted.narrow(1, 0, 1)
                pre_target = torch.cat((pre_target, temp.cpu()), 0)
                true_target = torch.cat((true_target, target.cpu()), 0)

        score_precision, score_recall, score_f1, _ = precision_recall_fscore_support(true_target, pre_target, average='macro')
        plot_x.append(epoch + 1)
        plot_acc.append(correct_vali / total_vali)
        plot_f1.append(score_f1)
        plot_precision.append(score_precision)
        plot_recall.append(score_recall)

        print(f'Epoch {epoch+1} - Time: {epoch_time:.2f}s, Accuracy: {100 * correct_vali / total_vali:.4f}%, Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), './model&img/model.pt')

    # Plot epoch times
    plt.figure()
    plt.plot(range(1, EPOCH + 1), epoch_times, label="Epoch Time (s)")
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.title("Epoch Duration over Training")
    plt.legend()
    plt.savefig('./model&img/epoch_times.png')

    # Plot other metrics
    plotScore(plot_x, plot_acc, plot_recall, plot_precision, plot_f1, plot_x_loss, plot_loss)

