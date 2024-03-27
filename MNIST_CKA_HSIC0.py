from MNIST_framework_new import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
import os
import numpy as np
import pickle
from scipy.spatial.distance import cosine
import random
import math
import time
from tqdm import tqdm

learningRate = 1e-07
CKA_index = [2,3,4,5]
selected_layers = '2345'
modelName = 'mnist10_epoch_15'

torch.set_default_tensor_type(torch.DoubleTensor)
model_init = torch.load(f'{modelName}.pt').double() # Try with model with 98% acc

import wandb
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="capstone-project-cka",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": learningRate,
    "architecture": "Net10",
    "dataset": "MNIST",
    "model_name" : modelName,
    "epochs": 100,
    "W1": 0,
    "W2": 1,
    "scheduler": "On",
    "selected_layers_CKA": selected_layers
    }
)

class CKA_Loss(torch.nn.Module):
    def __init__(self):
        super(CKA_Loss,self).__init__()

    def HSIC(self, K, L, device = "cuda"):
        """
        Computes the unbiased estimate of HSIC metric.
        """
        M = K.shape[0]
        I = torch.eye(M).to(device)
        unit = torch.ones([M, M]).to(device)
        H = I - unit/M
        Khat = H @ K @ H
        Lhat = H @ L @ H
        return torch.abs(torch.sum(Khat * Lhat))

    def forward(self, output, device = "cuda", test = False):
        M = len(output)
        hsic_matrix = torch.zeros(M, M)
        for i in range(M):
            X = output[i].flatten(1)
            K = X @ X.t()
            K.fill_diagonal_(0.0)
            hsic_k = self.HSIC(K, K)
            for j in range(i):
                Y = output[j].flatten(1)
                L = Y @ Y.t()
                L.fill_diagonal_(0.0)
                assert K.shape == L.shape, f"Feature shape mismatch! {K.shape}, {L.shape}"
                hsic_matrix[i, j] = self.HSIC(K, L) / torch.sqrt(hsic_k *self.HSIC(L, L)+1e-06)
        # absolute sum of differ
        id = torch.eye(M)
        l = torch.sum(torch.abs(hsic_matrix))
        hsic_visual = hsic_matrix + torch.transpose(hsic_matrix,0,1) +id
        # only adjacent cells
#         loss_area = torch.diagonal(hsic_matrix, 1)
#         l = torch.sum(torch.abs(loss_area))
        return hsic_visual,l
    
# Training settings
torch.manual_seed(1122)
device = torch.device("cuda")
train_kwargs = {'batch_size': 64}
test_kwargs = {'batch_size': 64}
cuda_kwargs = {'num_workers': 1,'pin_memory': True,'shuffle': True}
train_kwargs.update(cuda_kwargs)
test_kwargs.update(cuda_kwargs)
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('../data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

import torch
from torchvision import datasets, transforms

def add_white_noise(img):
    """
    Adds white noise to an image. 
    The noise has zero mean and a standard deviation of 0.4
    """
    noise = torch.randn(img.size()) * 0.4
    noisy_img = img + noise
    return noisy_img

transform_ood = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    add_white_noise  # Add your custom transform here
])

dataset1_ood = datasets.MNIST('../data', train=True, download=True, transform=transform_ood)
train_loader_ood = torch.utils.data.DataLoader(dataset1_ood, **train_kwargs)

dataset2_ood = datasets.MNIST('../data', train=False, download=True, transform=transform_ood)
test_loader_ood = torch.utils.data.DataLoader(dataset2_ood, **test_kwargs)

# def train(model, device, train_loader, optimizer, CKA_index, w1, w2):
def train(model, device, train_loader, optimizer, epoch, w1, w2):
    model.train()
    history = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data) # output is a length-10 tuple for 10 layers outputs
        optimizer.zero_grad()
        criterion = CKA_Loss()
        _,CKALoss = criterion([output[idx] for idx in CKA_index], device)
        loss = CKALoss
        # _,CKALoss = criterion(output[:10], device)
        # originalLoss = F.nll_loss(output[-1], target)
        # loss = w1*originalLoss + w2*CKALoss
        history.append(loss)
        if torch.isnan(loss):
            break
        loss.backward()
        optimizer.step()
    print('Train Epoch: {}  Loss: {:.6f}'.format(epoch, loss.item()))
    return history

def plotHeatmap(heatmap, index, epoch, testDataLabel, acc):
    plt.imshow(np.flipud(heatmap), cmap='magma', interpolation='nearest',vmin=0.0,vmax=1.0)
    plt.colorbar(label='Similarity')
    plt.title(f'CKA - Epoch{epoch} - {testDataLabel} - {acc}')
    plt.xlabel('Layer')
    plt.ylabel('Layer')
    plt.xticks(ticks=np.arange(0, heatmap.shape[0]), labels=index)
    plt.yticks(ticks=np.arange(heatmap.shape[0]-1,-1,-1), labels=index)
    plt.tight_layout()
    plt.savefig(f'CKA Heatmaps_CKA only/selectedLayers/CKA Heatmap - Epoch {epoch} - {testDataLabel} - {selected_layers}.png')
    plt.show()
    plt.clf()

# def test(model, device, test_loader,CKA_index):
def test(model, device, test_loader, epoch, testDataLabel):
    model.eval()
    nll_loss = 0
    cka_loss = 0
    cka_loss_p = 0
    correct = 0
    m_p = np.zeros((len(CKA_index),len(CKA_index)))
    m = np.zeros((10, 10))
    model.cuda()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
#             print(len(output))
            nll_loss += F.nll_loss(output[-1], target, reduction='sum').item()  # sum up batch loss
            criterion = CKA_Loss()

            # for all layers
            matrix, loss= criterion(output[:10], device)
            cka_loss += loss.item()
            m += matrix.detach().numpy().reshape((10,10))
            
            # for partial layers
            matrix_p, loss_p = criterion([output[idx] for idx in CKA_index], device)
            cka_loss_p += loss_p.item()
            m_p += matrix_p.detach().numpy().reshape((len(CKA_index),len(CKA_index)))
            
            pred = output[-1].argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    nll_loss /= len(test_loader)
    cka_loss /= len(test_loader)
    cka_loss_p /= len(test_loader)
    m /= len(test_loader)
    # m_p /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)
    plotHeatmap(m, list(range(10)), epoch, testDataLabel, acc)
    print(f'{testDataLabel} Result:')
    print('\nTest set: Average NLL loss: {:.4f}, Average CKA loss: {:.4f}, {:.4f} (selected layers), Accuracy: {}/{} ({:.0f}%)\n'.format(
        nll_loss, cka_loss, cka_loss_p, correct, len(test_loader.dataset),  acc))
    return nll_loss, cka_loss, acc


optimizer = optim.Adadelta(model_init.parameters(), lr=learningRate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
loss = []
loss_ood = []
acc = []
acc_ood = []
history_epoch = []
time_batch = 0
best_acc = -1
for epoch in range(1, 100):
    print(f'Epoch: {epoch}')
    start_time = time.time()
    history = train(model_init, device, train_loader, optimizer, epoch, 1, 1)
    assert len(history) == len(train_loader), "Nan with batch {}".format(len(history))  
    nll_l, cka_l, a = test(model_init, device, test_loader, epoch, 'In-Distribution')
    nll_l_ood, cka_l_ood, a_ood = test(model_init, device, test_loader_ood, epoch, 'OOD')
    loss.append(cka_l)
    acc.append(a)
    loss_ood.append(cka_l_ood)
    acc_ood.append(a_ood)
    scheduler.step()
    history_epoch.append(history)
    time_batch += time.time()-start_time
    wandb.log({"epoch": epoch, "acc":a, "loss": cka_l, "acc_ood": a_ood, "loss_ood": cka_l_ood, "nll_loss": nll_l, "nll_loss_ood":nll_l_ood})
    if a > best_acc:
        best_acc = a
        torch.save(model_init.state_dict(), f"best_model_selected_layers_{selected_layers}.pth")
print(time_batch/3)
wandb.finish()
print('Final Printed Results')
print('------------------------')
print('l')
print(loss)
print('l_ood')
print(loss_ood)
print('acc')
print(acc)
print('acc_ood')
print(acc_ood)

def plot_loss_trend(storedList, testDataLabel, yAxisLabel):
    epochs = list(range(1, len(storedList) + 1))  # Assuming epochs start from 1
    plt.plot(epochs, storedList, label=f'{yAxisLabel} per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(f'{yAxisLabel}')
    plt.title('Loss Trend over Epochs')
    plt.legend()
    plt.show()
    plt.clf()