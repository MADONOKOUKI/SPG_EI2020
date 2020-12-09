from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import lpips
import random
from torchvision.utils import save_image
from tanaka_adaptation_network import ShakePyramidNet
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim
import os
from utils.learnable_encryption import BlockScramble

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

parser.add_argument("--key_file_name", type=str, default="LE")

# For Networks
parser.add_argument("--depth", type=int, default=26)
parser.add_argument("--w_base", type=int, default=64)
parser.add_argument("--cardinary", type=int, default=4)

parser.add_argument("--save_img_directory", type=str, default="file")
parser.add_argument("--key_file_name", type=str, default="LE")

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
def train(epoch, name):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    p = None
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        imgs = inputs.numpy().astype('float32')
        bs = BlockScramble(name)
        imgs = np.transpose(imgs,(0 ,2 ,3 ,1 ))
        imgs = bs.Scramble(imgs)
        inputs = torch.from_numpy(np.transpose(imgs,(0,3,1,2)))
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

def test(epoch, name):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            imgs = inputs.numpy().astype('float32')
            bs = BlockScramble( name )
            imgs = np.transpose(imgs,(0 ,2 ,3 ,1 ))
            imgs = bs.Scramble(imgs)
            # block scrambling
            inputs = torch.from_numpy(np.transpose(imgs,(0,3,1,2)))
            if batch_idx == 0:
                save_image(inputs[0:16],args.save_img_directory + "/pixel_based_paper.png",nrow=4,normalize=True)

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs, _ = net(inputs)
            true_loss = criterion(outputs, targets) 

            test_loss += true_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    return test_loss, 100.*correct/total

def test_lpips(epoch,name,flag):
    lpips_score, total = 0, 0
    paramset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform_test)
    paramloader = torch.utils.data.DataLoader(paramset, batch_size=256, shuffle=True, num_workers=16)
    for batch_idx, (inputs, targets) in enumerate(paramloader):
            imgs = inputs.numpy().astype('float32')
            bs = BlockScramble( name )
            imgs = np.transpose(imgs,(0 ,2 ,3 ,1 ))
            imgs = bs.Scramble(imgs)
            # block scrambling
            img = torch.from_numpy(np.transpose(imgs,(0,3,1,2)))

            if batch_idx == 0 and flag:
              for i in range(16):
                print(LPIPS.forward(img[i].view(1,3,32,32), inputs[i].view(1,3,32,32)).item())
            lpips_score += torch.sum(LPIPS.forward(img, inputs)).item()
            total += targets.size(0)
    return lpips_score / total

if __name__ == '__main__':
    args = parser.parse_args()
    score_max = 0
    selected_key_file = ""
    best_score = 0
    for tmp in range(5):
      key_file = "LE/"+str(args.key_file_name) + str(tmp)
      bs = BlockScramble( [4,4,3] )
      bs.save(key_file)
      file_name = key_file
      score = test_lpips(0,key_file, True)
      if best_score < score:
        best_score = score
        selected_key_file = key_file 
    test_lpips(0, selected_key_file, True)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
            milestones=[int(e) for e in args.milestones.split(',')])
    best_acc = 0
    for epoch in range(0, 100):
      scheduler.step()
      train_loss, train_acc = train(epoch+1, selected_key_file)
      test_loss, test_acc = test(epoch+1, selected_key_file)
      print(train_acc, test_acc)
      if test_acc > best_acc:
        best_acc = test_acc
    print(best_acc)
