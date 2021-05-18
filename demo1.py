from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
import soft_renderer as sr

from loss import BFMFaceLoss
from dataset import CACDDataset
from model import BaseModel

import h5py

import trimesh
import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imsave


BATCH_SIZE = 1
MODEL_LOAD_PATH="./model_result_full/epoch_15_loss_0.7787_Img_loss_0.0112_LMK_loss0.7674.pth"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
# train_set = CACDDataset("./data/CACD2000_train.hdf5", val_transform, inv_normalize)
val_set = CACDDataset("./data/CACD2000_test.hdf5", val_transform, inv_normalize)

# train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

base_model = BaseModel(IF_PRETRAINED=True)
base_model.to(device)
base_model.load_state_dict(torch.load(MODEL_LOAD_PATH)['model'])
base_model.eval()

# ------------------------- Loss loading --------------------------------
camera_distance = 2.732
elevation = 0
azimuth = 0

renderer = sr.SoftRenderer(image_size=250, sigma_val=1e-4, aggr_func_rgb='hard', 
                            camera_mode='look_at', viewing_angle=30, fill_back=False,
                            perspective=False, light_intensity_ambient=1.0, light_intensity_directionals=0)

renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
face_loss = BFMFaceLoss(renderer, 1, 20, device)

for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
    in_img, gt_img, _ = data
    in_img = in_img.to(device)
    with torch.no_grad():
        recon_params = base_model(in_img)
        recon_img, shape, tri, albedo = face_loss.reconst_img(recon_params, "all")
    recon_img = recon_img.permute(0,2,3,1).cpu().numpy()
    recon_params =recon_params.cpu().numpy()

    for j in range(5,recon_img.shape[0]):
        
        mesh = trimesh.Trimesh(vertices=shape[j].cpu().numpy(), 
                               faces=tri[j].cpu().numpy(), 
                               vertex_colors=np.clip(albedo[j].cpu().numpy(),0,1))
        mesh.export("example.ply")

        fig = plt.figure(figsize=(300, 300))
        fig.add_subplot(1,2,1)
        plt.imshow(gt_img[j].permute(1,2,0).numpy())
        fig.add_subplot(1,2,2)
        plt.imshow(recon_img[j])
        plt.show()
        fig.savefig("./demo_output/result_demo1"+str(i)+".png")
