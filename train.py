import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms

import numpy as np
from tqdm import tqdm
from skimage import io
import soft_renderer as sr

from dataset import CACDDataset
from model import BaseModel
from loss import BFMFaceLoss

# -------------------------- Hyperparameter ------------------------------
BATCH_SIZE=32
NUM_EPOCH=15
VERBOSE_STEP=50
LR=3e-5
VIS_BATCH_IDX=7
LMK_LOSS_WEIGHT=1
RECOG_LOSS_WEIGHT=20
# MODEL_LOAD_PATH="./model_result_full/epoch_04_loss_15.5574_lmk_loss_0.0120_img_loss0.7773.pth"
MODEL_LOAD_PATH=None
SEED=0

# -------------------------- Reproducibility ------------------------------
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)
# print(torch.cuda.is_available())

train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

val_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

inv_normalize = transforms.Compose([
                    transforms.Normalize(
                                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                std=[1/0.229, 1/0.224, 1/0.255])
    ])
                    

# -------------------------- Dataset loading -----------------------------
train_set = CACDDataset("./data/CACD2000_train.hdf5", train_transform, inv_normalize)
val_set = CACDDataset("./data/CACD2000_val.hdf5", val_transform, inv_normalize)

# print("input_img size: ", train_set[0][0].shape)
# print("target_img size: ", train_set[0][1].shape)
# print("landmark size: ", train_set[0][2].shape)

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)


# -------------------------- Model loading ------------------------------
model = BaseModel(IF_PRETRAINED=True)
model.to(device)
if MODEL_LOAD_PATH is not None:
    model.load_state_dict(torch.load(MODEL_LOAD_PATH)['model'])

# -------------------------- Optimizer loading --------------------------
optimizer = optim.Adam(model.parameters(), lr=LR)
lr_schduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5)
if MODEL_LOAD_PATH is not None:
    optimizer.load_state_dict(torch.load(MODEL_LOAD_PATH)['optimizer'])

# ------------------------- Loss loading --------------------------------
camera_distance = 2.732
elevation = 0
azimuth = 0

renderer = sr.SoftRenderer(image_size=224, sigma_val=1e-4, aggr_func_rgb='hard', 
                            camera_mode='look_at', viewing_angle=30, fill_back=False,
                            perspective=False, light_intensity_ambient=1.0, light_intensity_directionals=0)

renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
face_loss = BFMFaceLoss(renderer, LMK_LOSS_WEIGHT, RECOG_LOSS_WEIGHT, device)

# ------------------------- plot visualization --------------------------
def visualize_batch(gt_imgs, recon_imgs):
    gt_imgs = gt_imgs.cpu()
    recon_imgs = recon_imgs.cpu()
    bs = gt_imgs.shape[0]
    num_cols = 8
    num_rows = int(bs/num_cols)

    canvas = np.zeros((num_rows*224, num_cols*224*2, 3))
    img_idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            gt_img = gt_imgs[img_idx].permute(1,2,0).numpy()
            # print("gt_img size: ", gt_img.shape)
            recon_img = recon_imgs[img_idx,:3,:,:].permute(1,2,0).numpy()
            # print("recon_img size: ", recon_img.shape)
            canvas[i*224:(i+1)*224, j*224*2:(j+1)*224*2-224, :3] = gt_img
            canvas[i*224:(i+1)*224, j*224*2+224:(j+1)*224*2, :4] = recon_img
            img_idx += 1
    return (np.clip(canvas,0,1)*255).astype(np.uint8)


# ------------------------- train ---------------------------------------
def train(model, epoch):
    model.train()
    running_loss = []
    running_img_loss = []
    running_lmk_loss = []
    running_recog_loss = []
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, data in loop:
        in_img, gt_img, lmk = data
        in_img = in_img.to(device); lmk = lmk.to(device)
        gt_img = gt_img.to(device)
        optimizer.zero_grad()
        recon_params = model(in_img)
        recon_params = recon_params.to(device)
        # print("gt_img size: ", gt_img.shape)
        # print("lmk size: ", lmk.shape)
        # loss,img_loss,lmk_loss,recog_loss,_ = face_loss(recon_params, gt_img, lmk)
        loss,img_loss,lmk_loss,_ = face_loss(recon_params, gt_img, lmk)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
        running_img_loss.append(img_loss.item())
        running_lmk_loss.append(lmk_loss.item())
        # running_recog_loss.append(recog_loss.item())
        loop.set_description("Loss: {:.6f}".format(np.mean(running_loss)))

        if i % VERBOSE_STEP == 0 and i!=0:
            print ("Epoch: {:02}/{:02} Progress: {:05}/{:05} Loss: {:.6f} Img Loss: {:.6f} LMK Loss: {:.6f} ".format(epoch+1, 
                                                                                                                    NUM_EPOCH, 
                                                                                                                    i, 
                                                                                                                    len(train_dataloader), 
                                                                                                                    np.mean(running_loss),
                                                                                                                    np.mean(running_img_loss),
                                                                                                                    np.mean(running_lmk_loss),
                                                                                                                    ))
            running_loss = []
            running_img_loss = []
            running_lmk_loss = []
            running_recog_loss = []

    return model

# ------------------------- eval ---------------------------------------
def eval(model, epoch):
    model.eval()
    all_loss_list = []
    img_loss_list = []
    lmk_loss_list = []
    recog_loss_list = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            in_img, gt_img, lmk = data
            in_img = in_img.to(device); lmk = lmk.to(device)
            gt_img = gt_img.to(device)

            recon_params = model(in_img)
            # import pdb; pdb.set_trace()
            # all_loss,img_loss,lmk_loss,recog_loss,recon_img=face_loss(recon_params, gt_img, lmk)
            all_loss,img_loss,lmk_loss,recon_img=face_loss(recon_params, gt_img, lmk)
            all_loss_list.append(all_loss.item())
            img_loss_list.append(img_loss.item())
            lmk_loss_list.append(lmk_loss.item())
            # recog_loss_list.append(recog_loss.item())
            if i == VIS_BATCH_IDX:
                visualize_image = visualize_batch(gt_img, recon_img)

    print ("-"*50, " Test Results ", "-"*50)
    _all_loss = np.mean(all_loss_list)
    _img_loss = np.mean(img_loss_list)
    _lmk_loss = np.mean(lmk_loss_list)
    # _recog_loss = np.mean(recog_loss_list)
    print ("Epoch {:02}/{:02} all_loss: {:.6f} image loss: {:.6f} landmark loss {:.6f}".format(epoch+1, NUM_EPOCH, _all_loss, _img_loss, _lmk_loss))
    print ("-"*116)
    # return _all_loss, _img_loss, _lmk_loss, _recog_loss, visualize_image
    return _all_loss, _img_loss, _lmk_loss, visualize_image

for epoch in range(0, NUM_EPOCH):
    model = train(model, epoch)
    # all_loss, img_loss, lmk_loss, recog_loss, visualize_image = eval(model, epoch)
    all_loss, img_loss, lmk_loss, visualize_image = eval(model, epoch)
    lr_schduler.step(all_loss)
    io.imsave("./result_full/Epoch:{:02}_AllLoss:{:.6f}_ImgLoss:{:.6f}_LMKLoss:{:.6f}.png".format(epoch, all_loss, img_loss, lmk_loss), visualize_image)
    model2save = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(model2save, "./model_result_full/epoch_{:02}_loss_{:.4f}_Img_loss_{:.4f}_LMK_loss{:.4f}.pth".format(epoch+1, img_loss+LMK_LOSS_WEIGHT*lmk_loss, img_loss, lmk_loss))

