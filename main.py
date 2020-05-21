import numpy as np
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import glob
from tqdm import tqdm
import logging
import os
from ez_transformers import *
# import torch
# from torch.utils.data import TensorDataset, DataLoader, random_split

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

def img_augmentation(images,save_name_prefix,img_augmentation_pre_image = 5,save_dir = 'data/augmentation_img'):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_dir += '/' + save_name_prefix
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order

    for i,img in enumerate(images):
        images_aug = seq(images=np.array([img for _ in range(img_augmentation_pre_image)]))
        for j,img_aug in enumerate(images_aug):
            grid_image = ia.draw_grid(np.array([img_aug]), cols=1)
            imageio.imwrite('%s/%s_i%d_j%d.jpg'%(save_dir,save_name_prefix,i,j),grid_image)
        

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
    print(Y.shape)
    print(X.shape)

    dataset = makeTorchDataset(X,Y)
    train_dataset,test_dataset = splitDataset(dataset)

