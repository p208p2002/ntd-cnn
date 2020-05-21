import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import os,sys
from datetime import datetime

def log(*logs):
    enablePrint()
    print(*logs)
    blockPrint()

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def saveModel(model,name):
    now = datetime.now()
    base_dir = 'train_models/'
    if(not os.path.isdir(base_dir)):
        os.mkdir(base_dir)
    save_dir = base_dir + now.strftime("%m-%d-%Y_%H-%M-%S_") + name
    os.mkdir(save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(save_dir)

def computeAccuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

def makeTorchDataset(*features):
    tensor_features = []
    for feature in features:
        tensor_feature = torch.tensor([f for f in feature],dtype=torch.float)
        tensor_features.append(tensor_feature)
    return TensorDataset(*tensor_features)

def splitDataset(full_dataset,split_rate = 0.8):
    train_size = int(split_rate * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset

def makeTorchDataLoader(torch_dataset,**options):
    #options: batch_size=int,shuffle=bool
    return DataLoader(torch_dataset,**options)
