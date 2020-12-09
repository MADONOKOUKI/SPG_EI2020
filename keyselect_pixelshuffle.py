from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from utils.pixel_based_encryption import pixel_based_encryption
import lpips
import random

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

def test_lpips(nagaposi, channel_shuffle):
    lpips_score, total = 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
            images = inputs.numpy().copy()
            img = pixel_based_encryption(images, nagaposi, channel_shuffle)
            # batch-wise evaluation
            lpips_score += torch.sum(criterion.forward(img, inputs)).item()
            # image-wise evaluation 
#            for i in range(img.size()[0]):
#              lpips_score += criterion.forward(img[i].view(1,3,32,32), inputs[i].view(1,3,32,32)).item()
            total += targets.size(0)
    return lpips_score / total

if __name__ == '__main__':
    args = parser.parse_args()
    inv = np.array([ np.random.randint(0, 2) for i in range(3072)])
    color = np.array([ np.random.randint(0, 6) for i in range(1024)])
    N_scores = []
    for rep_times in range(args.repeat_times+1):
      rep_times_check = 10
      score_max = 0
      N = []
      for rep in range(5):
        score_max = 0
        nagaposi_prop = []
        channel_shuffle_prop = []
        for tmp in range(rep_times_check):
          random.shuffle(inv)
          random.shuffle(color)
          tmp_score = test_lpips(inv, color)
          if score_max == 0:
            score_max = tmp_score
            nagaposi_prop = inv.copy()
            channel_shuffle_prop = color.copy()
          elif score_max <= tmp_score:
            score_max = tmp_score
            nagaposi_prop = inv.copy()
            channel_shuffle_prop = color.copy()
        N.append(score_max)
      print(score_max, N)
      N_scores.append(N)
    print(N_scores) 
