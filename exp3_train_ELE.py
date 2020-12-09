import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from shake_pyramidnet import ShakePyramidNet
import tensorboardX as tbx
import numpy as np
import torch
from scipy import linalg
from matplotlib.pyplot import imread
from torch.nn.functional import adaptive_avg_pool2d
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import math
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from itertools import permutations
import random
from utils.learnable_encryption import BlockScramble
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.utils import save_image
from proposed_adaptation_network import ShakePyramidNet
from util_norm import total_variation_norm
import lpips

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gamma', default=0.1, type=float)
parser.add_argument('--milestones', default='50, 75', type=str)

# For Networks
parser.add_argument("--depth", type=int, default=26)
parser.add_argument("--w_base", type=int, default=64)
parser.add_argument("--cardinary", type=int, default=4)

parser.add_argument("--save_img_directory", type=str, default="file")
parser.add_argument("--key_file_name", type=str, default="ELE")

# For Training
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--nesterov", type=bool, default=True)
parser.add_argument('--e', '-e', default=150, type=int, help='learning rate')
parser.add_argument("--batch_size", type=int, default=128)

args = parser.parse_args()

LPIPS = lpips.LPIPS(net='alex')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=16)

testset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=16)

net = ShakePyramidNet(depth=110, alpha=270, label=10)
net = net.to(device)

if device == 'cuda':
    print("true")
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay,
                nesterov=args.nesterov)

# Training
def train(epoch, name, best_idx, _shf):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    param12 = 0.001
    param3 = 0.001
    param4 = 1e-1
    p = None
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        x_stack = None

        images = inputs.numpy().copy()
        images = np.transpose(images,(0 ,2 ,3 ,1 ))
        for k in range(8):
          for j in range(8):
            key_file = str(name)+"/"+ str(best_idx) + "_" + str(k*8+j)+"_.pkl"
            bs = BlockScramble( key_file )
            images[:,k*4:(k+1)*4,j*4:(j+1)*4,:] = bs.Scramble(images[:,k*4:(k+1)*4,j*4:(j+1)*4,:])
        tmp = images.copy()
        for k in range(64):
            l = _shf[k]//8
            r = _shf[k]%8
            a = k//8
            b = k%8
            images[:,a*4:(a+1)*4,b*4:(b+1)*4,:] = tmp[:,l*4:(l+1)*4,r*4:(r+1)*4,:].copy()
        inputs = torch.from_numpy(np.transpose(images,(0 , 3, 1, 2)))

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, mat, feature = net(inputs)
        true_loss = criterion(outputs, targets) #+ 1e-1 * total_variation_norm(feature)
        # doubly stochastic constraint
        dsc = 0
        for i in range(64):
           dsc += torch.abs(mat[i,:]).sum()-torch.sqrt((mat[i,:]*mat[i,:]).sum())
           dsc += torch.abs(mat[:,i]).sum()-torch.sqrt((mat[:,i]*mat[:,i]).sum())

        dsc = param3 * dsc / (64*64)
        natural_image_prior = param4 * total_variation_norm(feature) / inputs.size()[0]

        loss = true_loss +  dsc + natural_image_prior

        loss.backward()
        optimizer.step()

        train_loss += true_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss, 100.*correct/total
def test(epoch, name, best_idx, _shf):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            x_stack = None
            images = inputs.numpy().copy()
            images = np.transpose(images,(0 ,2 ,3 ,1 ))
            for k in range(8):
              for j in range(8):
                key_file = str(name)+"/"+ str(best_idx) + "_" + str(k*8+j)+"_.pkl" 
                bs = BlockScramble( key_file )
                images[:,k*4:(k+1)*4,j*4:(j+1)*4,:] = bs.Scramble(images[:,k*4:(k+1)*4,j*4:(j+1)*4,:])
            tmp = images.copy()
            for k in range(64):
                l = _shf[k]//8
                r = _shf[k]%8
                a = k//8
                b = k%8
                images[:,a*4:(a+1)*4,b*4:(b+1)*4,:] = tmp[:,l*4:(l+1)*4,r*4:(r+1)*4,:].copy()
            inputs = torch.from_numpy(np.transpose(images,(0 , 3, 1, 2)))                               
            from torchvision.utils import save_image
            if batch_idx == 0:
              save_image(inputs[17],"paper_imgs/ele.png",nrow=1,normalize=True)
             # save_image(Variable(torch.from_numpy(np.transpose(images,(0,3, 1, 2))))[0:16].data, "icpr2020_exp2_2/"+ name +".png", nrow=4, normalize=True)              
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs,mat,feature = net(inputs)
            true_loss = criterion(outputs, targets) #+ 1e-1 * total_variation_norm(feature) / inputs.size()[0]

            test_loss += true_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    return test_loss, 100.*correct/total

def test_lpips(epoch,tem,best_idx,_shf):
    global best_acc
    mse_score, psnr_score, ssim_score, uqi_score, vif_score, brisque_score, niqe_score, unique_score, vgg_score, lpips_score = 0,0,0,0,0,0,0,0,0,0
    cnt, total = 0, 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
            images = inputs.numpy().copy()
            images = np.transpose(images,(0 ,2 ,3 ,1 ))
            for k in range(8):
              for j in range(8):
                key_file = str(name)+"/"+ str(best_idx) + "_" +str(k*8+j)+"_.pkl"
                bs = BlockScramble( key_file )
                images[:,k*4:(k+1)*4,j*4:(j+1)*4,:] = bs.Scramble(images[:,k*4:(k+1)*4,j*4:(j+1)*4,:])
            tmp = images.copy()
            for k in range(64):
                l = _shf[k]//8
                r = _shf[k]%8
                a = k//8
                b = k%8
                images[:,a*4:(a+1)*4,b*4:(b+1)*4,:] = tmp[:,l*4:(l+1)*4,r*4:(r+1)*4,:].copy()
            img = torch.from_numpy(np.transpose(images,(0 , 3, 1, 2)))

            lpips_score += torch.sum(LPIPS.forward(img, inputs)).item()
            inputs = img
            cnt = cnt + 1
            total += targets.size(0)
    return lpips_score / total

if __name__ == '__main__':
    args = parser.parse_args()
    cnts = []
    fids = []
    best_idx = 0
    best_score = 0
    rev_cnts = 32 * 32 * 3
    name = args.key_file_name
    file_name = name
    channel_shuffle_cnts = 32 * 32
    for tmp in range(1):
      for i in range(64):
        key_file = str(name)+"/"+ str(tmp) + "_" + str(i)+"_.pkl"
        bs = BlockScramble( [4,4,3] )
        bs.save(key_file)
      _shf = []
      for i in range(64):
        _shf.append(i)       
      random.shuffle(_shf)
      #file_name = str(idx) +"_" + str(idx_channel)
    score = test_lpips(0,0,tmp,_shf)
    print(score)
    if best_score < score:
        best_score = score
        best_idx = tmp
    scheduler = lr_scheduler.MultiStepLR(optimizer,
            milestones=[int(e) for e in args.milestones.split(',')])

    for epoch in range(0, 100):
        scheduler.step()
        train_loss, train_acc = train(epoch+1,name, best_idx,_shf)
        test_loss, test_acc = test(epoch+1,name,best_idx, _shf)
        print(str(train_loss / ((50000/512)+1)) +","+str(train_acc)+","+str(test_loss / ((10000/512)+1))+","+str(test_acc)+","+str(scheduler.get_lr()[0]))



