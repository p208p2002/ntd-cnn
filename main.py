import numpy as np
import imageio
import glob
from tqdm import tqdm
import logging
import os
from core import *
from cnn_model import Net
import torch
import torch.nn as nn
import torch.optim as optim
import time

FORMAT = 'line:%(lineno)d\t%(message)s'
logging.basicConfig(level=logging.INFO,format=FORMAT)
print = logging.info

ONE_HUNDRED_DOLLARS = '100'
FIVE_HUNDRED_DOLLARS = '500'
ONE_THOUSAND_DOLLARS = '1000'

def make_XY(data,label_ids):
    X = []
    Y = []
    for label_id,_X in zip(label_ids,data):
        #
        print(len(_X))
        y = [label_id]*len(_X)
        print(y)
        Y+=y
        #
        X += [x for x in _X]
        
    Y = np.array(Y)
    X = np.array(X)
    X = np.moveaxis(X, -1, 1) # move axis to fit pytorch input format -> N C H W
    print(Y)
    print(Y.shape)
    print(X.shape)
    time.sleep(1)
    return X,Y

def load_money_images(money_type,img_dir='data/money_img',split_test=False):
    print('load_money_images %s'%money_type)
    imgs = []
    img_paths = glob.glob(img_dir+'/'+money_type+'/*.jpg')
    # print(len(img_paths))
    # pbar = tqdm(total=len(img_paths))
    for img_path in img_paths:
        img = imageio.imread(img_path)
        imgs.append(img)
        # pbar.update(1)
    if(split_test):
        return np.array(imgs[:int(len(imgs)/2)]),np.array(imgs[int(len(imgs)/2):])
    else:
        return np.array(imgs)

def train(model,optimizer,loss_func,train_dataloader,device):
    model.train()
    for epoch in range(20):  # loop over the dataset multiple times
        running_loss_val = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_dataloader):
            data = tuple(t.to(device) for t in data)
            optimizer.zero_grad()
            outputs = model(data[0])

            loss = loss_func(outputs, data[-1])
            loss.backward()
            optimizer.step()

            # print statistics
            loss_t = loss.item()
            running_loss_val += (loss_t - running_loss_val) / (i + 1)

            acc_t = computeAccuracy(outputs, data[-1])
            running_acc += (acc_t - running_acc) / (i + 1)

            print('e:%d b:%d %3.5f %3.5f'%(epoch, i, running_loss_val, running_acc))
    print('Finished Training')
    return model

def test(model,test_dataloader,device):
    model.eval()
    running_acc = 0.0
    for i, data in enumerate(test_dataloader):
        data = tuple(t.to(device) for t in data)
        outputs = model(data[0])
        acc_t = computeAccuracy(outputs, data[-1])
        running_acc += (acc_t - running_acc) / (i + 1)
    
    print(running_acc)
    print('Finished Test')

if __name__ == "__main__":
    # 原始訓練資料
    one_hunderd_dollars, one_hunderd_dollars_test = load_money_images(ONE_HUNDRED_DOLLARS, img_dir='data/money_img', split_test=True)
    five_hunderd_dollars, five_hunderd_dollars_test = load_money_images(FIVE_HUNDRED_DOLLARS, img_dir='data/money_img', split_test=True)
    one_thousand_dollars, one_thousand_dollars_test = load_money_images(ONE_THOUSAND_DOLLARS, img_dir='data/money_img', split_test=True)
    
    # 資料增量
    os.system('rm -rf data/augmentation_img/')
    img_augmentation(one_hunderd_dollars, ONE_HUNDRED_DOLLARS)
    img_augmentation(five_hunderd_dollars, FIVE_HUNDRED_DOLLARS)
    img_augmentation(one_thousand_dollars, ONE_THOUSAND_DOLLARS)

    one_hunderd_dollars_augmentation = load_money_images(ONE_HUNDRED_DOLLARS, img_dir='data/augmentation_img')
    five_hunderd_dollars_augmentation = load_money_images(FIVE_HUNDRED_DOLLARS, img_dir='data/augmentation_img')
    one_thousand_dollars_augmentation = load_money_images(ONE_THOUSAND_DOLLARS, img_dir='data/augmentation_img')

    # 合併資料
    one_hunderd_dollars = np.concatenate([one_hunderd_dollars,one_hunderd_dollars_augmentation],axis=0)
    five_hunderd_dollars = np.concatenate([five_hunderd_dollars,five_hunderd_dollars_augmentation],axis=0)
    one_thousand_dollars = np.concatenate([one_thousand_dollars,one_thousand_dollars_augmentation],axis=0)

    # 準備訓練資料
    testing_data = np.array([one_hunderd_dollars_test,five_hunderd_dollars_test,one_thousand_dollars_test])
    traing_data = np.array([one_hunderd_dollars,five_hunderd_dollars,one_thousand_dollars])
    labels = [ONE_HUNDRED_DOLLARS,FIVE_HUNDRED_DOLLARS,ONE_THOUSAND_DOLLARS]
    label_ids = [i for i,_ in enumerate(labels)]
    print(label_ids)
    print(traing_data.shape)
    print(labels)
    print(label_ids)
    
    # 
    X,Y = make_XY(traing_data,label_ids)
    X_test,Y_test = make_XY(testing_data,label_ids)

    train_dataset = makeTorchDataset(X,Y)
    test_dataset = makeTorchDataset(X_test,Y_test)

    train_dataloader  =  makeTorchDataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader =  makeTorchDataLoader(test_dataset, batch_size=12, shuffle=True)
    
    # setting & init
    model = Net()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("using device %s"%device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=6e-5)

    # train
    train_model_options = {
        'model':model,
        'optimizer':optimizer,
        'loss_func':criterion,
        'train_dataloader':train_dataloader,
        'device':device
    }
    model = train(**train_model_options)

    # test
    test_model_options = {
        'model':model,
        'test_dataloader':test_dataloader,
        'device':device
    }
    test(**test_model_options)
