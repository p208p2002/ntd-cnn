import numpy as np
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import glob
from tqdm import tqdm
import logging
import os
from ez_transformers import *
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

def load_money_images(money_type,img_dir='data/money_img'):
    print('load_money_images %s'%money_type)
    imgs = []
    img_paths = glob.glob(img_dir+'/'+money_type+'/*.jpg')
    # print(len(img_paths))
    # pbar = tqdm(total=len(img_paths))
    for img_path in img_paths:
        img = imageio.imread(img_path)
        imgs.append(img)
        # pbar.update(1)
    return np.array(imgs)

def img_augmentation(images,save_name_prefix,img_augmentation_pre_image = 30,save_dir = 'data/augmentation_img'):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_dir += '/' + save_name_prefix
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # seq = iaa.Sequential([
    #     iaa.Fliplr(0.5), # horizontal flips
    #     iaa.Crop(percent=(0, 0.1)), # random crops
    #     iaa.Sometimes(
    #         0.5,
    #         iaa.GaussianBlur(sigma=(0, 0.5))
    #     ),
    #     iaa.LinearContrast((0.75, 1.5)),
    #     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    #     iaa.Multiply((0.8, 1.2), per_channel=0.2),
    #     iaa.Affine(
    #         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #         rotate=(-25, 25),
    #         shear=(-8, 8)
    #     ),
    #     iaa.Dropout(),
    #     iaa.Multiply()
    # ], random_order=True) # apply augmenters in random order

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.LinearContrast((0.5, 2.0))
                    )
                ]),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
            )
        ],
        random_order=True
    )

    for i,img in enumerate(images):
        images_aug = seq(images=np.array([img for _ in range(img_augmentation_pre_image)]))
        for j,img_aug in enumerate(images_aug):
            grid_image = ia.draw_grid(np.array([img_aug]), cols=1)
            imageio.imwrite('%s/%s_i%d_j%d.jpg'%(save_dir,save_name_prefix,i,j),grid_image)

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
    one_hunderd_dollars = load_money_images(ONE_HUNDRED_DOLLARS, img_dir='data/money_img')
    five_hunderd_dollars = load_money_images(FIVE_HUNDRED_DOLLARS, img_dir='data/money_img')
    one_thousand_dollars = load_money_images(ONE_THOUSAND_DOLLARS, img_dir='data/money_img')
    
    # 資料增量
    os.system('rm -rf data/augmentation_img/')
    img_augmentation(one_hunderd_dollars, ONE_HUNDRED_DOLLARS)
    img_augmentation(five_hunderd_dollars, FIVE_HUNDRED_DOLLARS)
    img_augmentation(one_thousand_dollars, ONE_THOUSAND_DOLLARS)

    one_hunderd_dollars_augmentation = load_money_images(ONE_HUNDRED_DOLLARS, img_dir='data/augmentation_img')
    five_hunderd_dollars_augmentation = load_money_images(FIVE_HUNDRED_DOLLARS, img_dir='data/augmentation_img')
    one_thousand_dollars_augmentation = load_money_images(ONE_THOUSAND_DOLLARS, img_dir='data/augmentation_img')

    one_hunderd_dollars = np.concatenate([one_hunderd_dollars,one_hunderd_dollars_augmentation],axis=0)
    five_hunderd_dollars = np.concatenate([five_hunderd_dollars,five_hunderd_dollars_augmentation],axis=0)
    one_thousand_dollars = np.concatenate([one_thousand_dollars,one_thousand_dollars_augmentation],axis=0)

    # 準備訓練資料
    traing_data = np.array([one_hunderd_dollars,five_hunderd_dollars,one_thousand_dollars])
    labels = [ONE_HUNDRED_DOLLARS,FIVE_HUNDRED_DOLLARS,ONE_THOUSAND_DOLLARS]
    label_ids = [i for i,_ in enumerate(labels)]
    print(label_ids)
    print(traing_data.shape)
    print(labels)
    print(label_ids)
    
    Y = []
    X = []
    for label_id,_X in zip(label_ids,traing_data):
        #
        print(len(_X))
        y = [label_id]*len(_X)
        print(y)
        Y+=y
        #
        X += [x for x in _X]
        
    Y = np.array(Y)
    X = np.array(X)
    X = np.moveaxis(X, -1, 1) # N C H W
    print(Y)
    print(Y.shape)
    print(X.shape)
    time.sleep(3)

    dataset = makeTorchDataset(X,Y)
    # train_dataset,test_dataset = splitDataset(dataset,split_rate=0.8)
    
    # train_dataloader =  makeTorchDataLoader(train_dataset, batch_size=3, shuffle=True)
    train_dataloader  =  makeTorchDataLoader(dataset, batch_size=32, shuffle=True)
    test_dataloader =  makeTorchDataLoader(dataset, batch_size=12, shuffle=True)
    

    
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
