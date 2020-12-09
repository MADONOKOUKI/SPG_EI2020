from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from utils.learnable_encryption import BlockScramble
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

def test_lpips(_shf):
    lpips_score, total = 0, 0
    key_file = "LE/"+str(0)+"_.pkl"
    bs = BlockScramble( key_file )
    for batch_idx, (inputs, targets) in enumerate(trainloader):
            images = inputs.numpy().copy()
            images = np.transpose(images,(0 ,2 ,3 ,1 ))
            images = bs.Scramble(images)
            img = torch.from_numpy(np.transpose(images,(0 , 3, 1, 2)))
            lpips_score += torch.sum(criterion.forward(img, inputs)).item()
            total += targets.size(0)
    return lpips_score / total

if not os.path.isdir("LE"):
  os.mkdir("LE")

if __name__ == '__main__':
    args = parser.parse_args()
    N_scores = []
    shuffle = []
    for i in range(64):
      shuffle.append(i)       
    for rep_times in range(args.repeat_times+1):
      score_max = 0
      N = []
      for rep in range(5):
        score_max = 0
        nagaposi_prop = []
        channel_shuffle_prop = []
        for tmp in range(rep_times):
          key_file = "LE/"+str(0)+"_.pkl"
          bs = BlockScramble( [4,4,3] )
          bs.save(key_file)
          random.shuffle(shuffle)
          tmp_score = test_lpips(shuffle)
          if score_max == 0:
            score_max = tmp_score
          elif score_max <= tmp_score:
            score_max = tmp_score
        N.append(score_max)
      print(score_max, N)
      N_scores.append(N)
    print(N_scores) 
