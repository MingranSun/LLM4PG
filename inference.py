import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_provider import Dataset_Pro9
import scipy.io as sio
# from models.pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction,PointNetFeaturePropagation
from models.GPT2_rgb_d_patching_fc import GPTRGBD, Conv_patching_RGB, Conv_patching_dep, Frequency_fc
import numpy as np
import shutil
import argparse
from torch.utils.tensorboard import SummaryWriter
import torchvision
import imageio
from tqdm import tqdm
from torchvision import transforms
import time

# from metrics import NMSELoss, SE_Loss


batch_size = 1
# device = torch.device('cuda:1')
# torch.cuda.set_device(2)
device = torch.device('cuda:0')
freq = 1.6

load_path_rgb = "weight/rgb_network.pth"
load_path_dep = "weight/dep_network.pth"
load_path_gpt = "weight/gpt_network.pth"
load_path_freq = "weight/freq_network.pth"



test_dep_path = "dataset/dep/"
test_RGB_path = "dataset/RGB/"
test_pl_path = "dataset/pl/"

test_set = Dataset_Pro9(test_dep_path, test_RGB_path, test_pl_path, transform=None)  # creat data for training

parser = argparse.ArgumentParser(description='GPTRGBD')
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--stride', type=int, default=64)
parser.add_argument('--seq_len', type=int, default=1024)

parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--freeze_inout', type=int, default=0)
parser.add_argument('--freeze_gptall', type=int, default=0)
parser.add_argument('--freeze_pointnet', type=int, default=0)

parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--pred_len', type=int, default=100)

args = parser.parse_args()
model_GPT = GPTRGBD(args, device).to(device)
model_CNN_RGB = Conv_patching_RGB(args, device).to(device)
model_CNN_dep = Conv_patching_dep(args, device).to(device)
model_freq = Frequency_fc(args, device).to(device)

model_CNN_dep = torch.load(load_path_dep, map_location=device, weights_only=False)
model_CNN_RGB = torch.load(load_path_rgb, map_location=device, weights_only=False)
model_GPT = torch.load(load_path_gpt, map_location=device, weights_only=False)
model_freq = torch.load(load_path_freq, map_location=device, weights_only=False)
# 加载参数
if os.path.exists(load_path_dep):
    model_CNN_dep = torch.load(load_path_dep, map_location=device, weights_only=False)
if os.path.exists(load_path_rgb):
    model_CNN_RGB = torch.load(load_path_rgb, map_location=device, weights_only=False)
if os.path.exists(load_path_gpt):
    model_GPT = torch.load(load_path_gpt, map_location=device, weights_only=False)
if os.path.exists(load_path_freq):
    model_freq = torch.load(load_path_freq, map_location=device, weights_only=False)

###################################################################
# ------------------- Main test (Run second)----------------------------------
###################################################################
def test(testing_data_loader):
    global total_loss
    print('Start testing...!!!!!!!!!!!')
    epoch_test_loss, epoch_test_loss2 = [], []
    progress_bar = tqdm(enumerate(testing_data_loader), total=len(testing_data_loader), desc=f'Testing', leave=True)

    model_CNN_dep.eval()
    model_CNN_RGB.eval()
    model_GPT.eval()
    model_freq.eval()
    with torch.no_grad():
        i = 0
        all_nmse = []
        for iteration, batch in enumerate(testing_data_loader, 1):
            
            
            prev_dep, prev_RGB, pred_t = Variable(batch[0]).to(device).float(), \
                    Variable(batch[1]).to(device).float(),\
                    Variable(batch[2]).to(device).float()
                #                optimizer.zero_grad()  # fixed
            start = time.time()
            dep_features = model_CNN_dep(prev_dep)
            rgb_features = model_CNN_RGB(prev_RGB)
            freq_features = model_freq(torch.tensor([freq]).to(device))
            freq_features = freq_features.unsqueeze(0).unsqueeze(0)
            batch_size = dep_features.shape[0]
            freq_features = freq_features.repeat(batch_size, 1, 1)
            rgb_d_features = torch.cat((dep_features, rgb_features, freq_features), dim = 1)

            
            pred_m = model_GPT(rgb_d_features)
            
            end = time.time()
            test_time = end - start
            print('test time: {:.7f}'.format(test_time))
            #print('pred_m',pred_m.shape)
            #print('pred_t',pred_t.shape)   
            #print("RGB",prev_RGB.shape) 
            #trans_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            #pl_image_3c = trans_norm(pl_pre)
            # pl_image_3c = pl_pre.repeat(1, 3, 1, 1)
            # pred_m : prediction
            pred_m_3c = pred_m.repeat(1, 3, 1, 1)
            pred_t_3c = pred_t.repeat(1, 3, 1, 1)
            pl_va = (pred_t - pred_t.min()) * (255 / (pred_t.max() - pred_t.min()))
            pl_va = pl_va.cpu()
            pl_pre_va = (pred_m - pred_m.min()) * (255 / (pred_m.max() - pred_m.min()))
            pl_pre_va = pl_pre_va.cpu()
            #print('pl_va',pl_va)
            #print('pl_pre_va',pl_pre_va)
            #print('pl_va-pl_pre_va',pl_va - pl_pre_va)
            error = torch.sum((pl_va - pl_pre_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            all_nmse.append(nmse)
            all_nmse_mean = np.mean(all_nmse)
            print("i",i)
            print(f"NMSE: {nmse}")
            print(f"NMSE_mean: {all_nmse_mean}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            prev_RGB,
                            prev_dep,
                            pred_m_3c,
                            pred_t_3c,
                           
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            i = i + 1
           
            progress_bar.set_postfix(loss=nmse.item())
            progress_bar.update()  # 更新进度条
        


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":

    testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=batch_size, shuffle=False,
                                     pin_memory=True,
                                     drop_last=True)  # put testing data to DataLoader for batches


    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss()
    test(testing_data_loader)  # call test function (
