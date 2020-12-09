from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from utils.etc_encryption import EtC_encryption
import lpips
import random
import os

criterion = lpips.LPIPS(net='alex')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = True if torch.cuda.is_available() else False

if cuda:
    criterion = torch.nn.DataParallel(criterion).cuda()
    cudnn.benchmark = True

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--repeat_times", type=int, default=20)

args = parser.parse_args()

# from cifar10 import CIFAR10
trainset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=16)

def test_lpips(sets):
    lpips_score, total = 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
            imgs = inputs.numpy().astype('float32')
            imgs = np.transpose(imgs,(0 ,2 ,3 ,1 ))
            img = EtC_encryption(imgs, sets)
            lpips_score += torch.sum(criterion.forward(img, inputs)).item()
            total += targets.size(0)
    return lpips_score / total

if __name__ == '__main__':
    args = parser.parse_args()
    N_scores = []
    _rotate = []
    _negaposi = []
    _reverse = []
    _channel = []
    _shf = []
    for i in range(64):
      x = random.randint(0,3)
      z = random.randint(0,2)
      a = random.randint(0,5)
      if i%2==0:
          _negaposi.append(1)
      else:
          _negaposi.append(0)
      _rotate.append(x)
      _reverse.append(z)
      _channel.append(a)
      _shf.append(i)
    for rep_times in range(args.repeat_times+1):
      score_max = 0
      N = []
      for rep in range(5):
        score_max = 0
        for tmp in range(rep_times):
          random.shuffle(_rotate)
          random.shuffle(_negaposi)
          random.shuffle(_reverse)
          random.shuffle(_channel)
          random.shuffle(_shf)
          sets = [_rotate, _negaposi, _reverse, _channel, _shf]
          tmp_score = test_lpips(sets)
          if score_max == 0:
            score_max = tmp_score
          elif score_max <= tmp_score:
            score_max = tmp_score
        N.append(score_max)
      print(score_max, N)
      N_scores.append(N)
    print(N_scores) 
