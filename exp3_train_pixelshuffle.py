from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from utils.pixel_based_encryption import pixel_based_encryption
import lpips
import random
from torchvision.utils import save_image
from shake_pyramidnet_adap import ShakePyramidNet
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim
import os

LPIPS = lpips.LPIPS(net='alex')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = True if torch.cuda.is_available() else False

if cuda:
    LPIPS = torch.nn.DataParallel(LPIPS).cuda()
    cudnn.benchmark = True

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--repeat_times", type=int, default=20)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gamma', default=0.1, type=float)
parser.add_argument('--milestones', default='50,75', type=str)

# For Networks
parser.add_argument("--depth", type=int, default=26)
parser.add_argument("--w_base", type=int, default=64)
parser.add_argument("--cardinary", type=int, default=4)

parser.add_argument("--save_img_directory", type=str, default="file")

# For Training
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--nesterov", type=bool, default=True)
parser.add_argument('--e', '-e', default=150, type=int, help='learning rate')
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--div_bit", type=float, default=128)
parser.add_argument("--inv_ratio", type=float, default=0.5)
       #rser.add_argument("--batch_size", type=int, default=128) print(self.blockSize)
args = parser.parse_args()

if not os.path.isdir(args.save_img_directory):
  os.mkdir(args.save_img_directory)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# from cifar10 import CIFAR10
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
cuda = True if torch.cuda.is_available() else False

optimizer = optim.SGD(net.parameters(),
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay,
                nesterov=args.nesterov)
# Training
def train(epoch, nagaposi, channel_shuffle):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    p = None
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        images = inputs.numpy().copy()
        inputs = pixel_based_encryption(images, nagaposi, channel_shuffle)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs, outputs2 = net(inputs)
        loss = criterion(outputs, targets) #+ 1e-1 * total_variation_norm(feature)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss, 100.*correct/total

def test(epoch, nagaposi, channel_shuffle):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            images = inputs.numpy().copy()
            inputs = pixel_based_encryption(images, nagaposi, channel_shuffle)

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if batch_idx == 0 and epoch == 100:
                save_image(inputs[0:16],args.save_img_directory + "/pixel_based_paper.png",nrow=4,normalize=True)
                for i in range(16):
                  print(inputs.size())
                  print(net(inputs[i,:,:,:].view(1,3,32,32))[0])

            outputs, _ = net(inputs)
            true_loss = criterion(outputs, targets) 

            test_loss += true_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    return test_loss, 100.*correct/total

def test_lpips(nagaposi, channel_shuffle, flag):
    lpips_score, total = 0, 0
    paramset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform_test)
    paramloader = torch.utils.data.DataLoader(paramset, batch_size=256, shuffle=True, num_workers=16)
    for batch_idx, (inputs, targets) in enumerate(paramloader):
            images = inputs.numpy().copy()
            img = pixel_based_encryption(images, nagaposi, channel_shuffle)
            if batch_idx == 0 and flag:
              for i in range(16):
                print(LPIPS.forward(img[i].view(1,3,32,32), inputs[i].view(1,3,32,32)).item())
            lpips_score += torch.sum(LPIPS.forward(img, inputs)).item()
            total += targets.size(0)
    return lpips_score / total

if __name__ == '__main__':
    args = parser.parse_args()
    inv = np.array([ np.random.randint(0, 2) for i in range(3072)])
    color = np.array([ np.random.randint(0, 6) for i in range(1024)])
    N_scores = []
    score_max = 0
    N = []
    for rep in range(10): # trial number 
      score_max = 0
      nagaposi_prop = []
      channel_shuffle_prop = []
      for tmp in range(1):
        random.shuffle(inv)
        random.shuffle(color)
        tmp_score = test_lpips(inv, color, False)
        if score_max == 0:
          score_max = tmp_score
          nagaposi_prop = inv.copy()
          channel_shuffle_prop = color.copy()
        elif score_max <= tmp_score:
          score_max = tmp_score
          nagaposi_prop = inv.copy()
          channel_shuffle_prop = color.copy()
      N.append(tmp_score)
    #print(score_max, N)
    print(N)
    print(score_max)
    test_lpips(nagaposi_prop, channel_shuffle_prop, True)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
            milestones=[int(e) for e in args.milestones.split(',')])
    best_acc = 0
    for epoch in range(0, 100):
      scheduler.step()
      train_loss, train_acc = train(epoch+1, nagaposi_prop, channel_shuffle_prop)
      test_loss, test_acc = test(epoch+1, nagaposi_prop, channel_shuffle_prop)
      print(train_acc, test_acc)
      if test_acc > best_acc:
        best_acc = test_acc
    print(best_acc)
