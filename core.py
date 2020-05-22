import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import os,sys
from datetime import datetime
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import imageio
import glob
import logging
FORMAT = '%(filename)s line:%(lineno)d\t%(message)s'
logging.basicConfig(level=logging.INFO,format=FORMAT)
print = logging.info

def make_XY(data,label_ids):
    X = []
    Y = []
    for label_id,_X in zip(label_ids,data):
        #
        # print(len(_X))
        y = [label_id]*len(_X)
        # print(y)
        Y+=y
        #
        X += [x for x in _X]
        
    Y = np.array(Y)
    X = np.array(X)
    X = np.moveaxis(X, -1, 1) # move axis to fit pytorch input format -> N C H W
    print(Y)
    print(Y.shape)
    print(X.shape)
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
    for i,feature in enumerate(features):
        if(i+1 == len(features)):
            tensor_feature = torch.tensor([f for f in feature],dtype=torch.long)
        else:
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

def img_augmentation(images,save_name_prefix,img_augmentation_pre_image = 60,save_dir = 'data/augmentation_img'):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_dir += '/' + save_name_prefix
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

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