import numpy as np
import imageio
import logging
import os
from core import *
from cnn_model import Net
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

FORMAT = '%(filename)s line:%(lineno)d\t%(message)s'
logging.basicConfig(level=logging.INFO,format=FORMAT)
print = logging.info

ONE_HUNDRED_DOLLARS = '100'
FIVE_HUNDRED_DOLLARS = '500'
ONE_THOUSAND_DOLLARS = '1000'

traing_loss_rec = []
traing_acc_rec = []

def train(model,optimizer,loss_func,train_dataloader,device):
    model.train()
    for epoch in range(100):  # loop over the dataset multiple times
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
        
        #
        print('e:%d %3.5f %3.5f'%(epoch, running_loss_val, running_acc))

        #
        traing_loss_rec.append(running_loss_val)
        traing_acc_rec.append(running_acc)

        #
        # if(epoch>=3):
        #     break
        if(running_loss_val <= 0.35 and epoch >= 50):
            print('early stop')
            break
           
    print('Finished Training')
    return model

def test(model,test_dataloader):
    model = model.to('cpu')
    model.eval()
    running_acc = 0.0
    y_true,y_pred=[],[]

    for i, data in enumerate(test_dataloader):
        data = tuple(t.to('cpu') for t in data)
        outputs = model(data[0])
        acc_t = computeAccuracy(outputs, data[-1])
        running_acc += (acc_t - running_acc) / (i + 1)

        #
        outputs = outputs.squeeze().detach().numpy()
        output_y = np.argmax(outputs)
        label_y = data[-1].squeeze().detach().numpy()
        y_pred.append(output_y)
        y_true.append(label_y)
        
    
    print('test acc:%f'%running_acc)
    print('Finished Test')
    return y_pred,y_true

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
    test_dataloader =  makeTorchDataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # setting & init
    model = Net()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("using device %s"%device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-6)

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
    }
    y_pred,y_true = test(**test_model_options)

    #
    plt.plot([i for i in range(len(traing_acc_rec))],traing_acc_rec,'g',label='acc')
    plt.legend(loc='best')
    plt.savefig('acc_score')
    plt.clf()
    
    #
    plt.plot([i for i in range(len(traing_loss_rec))],traing_loss_rec,'r',label='loss')
    plt.legend(loc='best')
    plt.savefig('loss_score')
    plt.clf()

    #
    cm_array = confusion_matrix(y_true, y_pred)
    print('confusion_matrix')
    print(cm_array)
    df_cm = pd.DataFrame(cm_array, [l for l in labels], [l for l in labels])
    plt.figure(figsize = (len(labels),len(labels)))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}) # font size
    plt.savefig('confusion_matrix')
    plt.clf()
