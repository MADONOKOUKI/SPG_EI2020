from exp1_fid_score_only_nagapos import calculate_fid_given_paths
from exp1_fid_score_only_nagapos_normalized import calculate_normalized_fid_given_paths
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
from fid_score_negative_pixelchange_lap import calculate_fid_given_paths_lap
from vgg_perceptual_loss import VGGPerceptualLoss
from inception import InceptionV3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('--num_of_repeat', type=int, default=1)
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=False, num_workers=16)

if device == 'cuda':
    print("true")
    cudnn.benchmark = True


def negaposi(val,o):
  out = val
  p = np.full((val.shape[0],),255)
  if o == 0:
    out = p - val
  return out

import pytorch_ssim
import lpips
import pytorch_msssim
from full_ref import psnr
ssim_loss = pytorch_ssim.SSIM()
ms_ssim_loss = pytorch_msssim.MSSSIM()
mseloss = nn.MSELoss()
vgg_loss = VGGPerceptualLoss()
criterion = lpips.LPIPS(net='alex')

def test(epoch,tem,idx, num):
    global best_acc
    #idx_name = idx.copy()
    mse_score, psnr_score, ssim_score, uqi_score, vif_score, brisque_score, niqe_score, unique_score, vgg_score, lpips_score = 0,0,0,0,0,0,0,0,0,0
    cnt, total = 0, 0
    rev = (np.asarray(idx) > len(idx)/2) 
    for batch_idx, (inputs, targets) in enumerate(trainloader):
            img = inputs[:,:,:,:].numpy().astype('float32')

            img = np.transpose(img,(0,2, 3, 1))
            images = img.astype('float32') * 255
            for i in range(32):
              for j in range(32):
                for k in range(3):
                # negaposi
                  images[:,i,j,k] = negaposi(images[:,i,j,k].copy(), idx[i*96+j*3+k])
            img = images / 255
            img = torch.from_numpy(np.transpose(img,(0,3, 1, 2)))

            from torchvision.utils import save_image
            if batch_idx == 0:
              for idx_img in range(16):
                print(idx_img, torch.sum(criterion.forward(img[idx_img], inputs[idx_img])).item())
              save_image(img[0:16],"paper_imgs/"+str(tem)+"_"+str(tem)+"_"+str(j)+"_preexperiment.png",nrow=4,normalize=True)
            # MSE
            mse_score += mseloss(img*255,255*inputs).item()
            # SSIM
            ssim_score += 255 * ssim_loss(img,inputs).item()
            # PSNR 
            for cnt in range(img.size()[0]):
              psnr_score += psnr(np.transpose((255*img).numpy().astype('int32')[cnt,:,:,:] ,(1, 2, 0)) , np.transpose((255*inputs).numpy().astype('int32')[cnt,:,:,:],(1, 2, 0)))
            # Perceptual SImilairty cvpr 2018 https://github.com/richzhang/PerceptualSimilarity#1-learned-perceptual-image-patch-similarity-lpips-metric
            lpips_score += torch.sum(criterion.forward(img, inputs)).item()
            # vgg_score
            vgg_score += vgg_loss(img*255,255*inputs).item()

            inputs = img
            cnt = cnt + 1
            total += targets.size(0)
    return mse_score / total, psnr_score / total, ssim_score / total, brisque_score / total, niqe_score / total, unique_score / total, vgg_score / total, lpips_score / total
#    return mse / total, psnr_score / total, ssim_score / total, ms_ssim_score / total, vif_score / total, gmsd_score / total, uqi_score / total, persim_dist / total#, brisque / total, unique / total

if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='data/', download=True,
                             train = False,
                             transform=transforms.Compose([
                                 transforms.Scale(32),
                                 transforms.ToTensor(),
                             ])
    )
    from itertools import permutations
    import random
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    random.seed(30)
    IgnoreLabelDataset(cifar)
    cnts, mses, psnrs, ssims, uqi_scores, vif_scores, brisques, niqes, uniques, vggs, lpipses, fids =[],[],[],[],[],[],[],[],[],[],[],[]
    rev_cnts = 32 * 32 * 3
    rev_simluations = [0, 307, 614, 921, 1228, 1536, 1843, 2150, 2457, 2764, rev_cnts]
    all_sets = []
    for idx in rev_simluations:
      print("shuffle counts",idx)
      p = []
      for i in range(idx):
        p.append(0)
      for i in range(rev_cnts-idx):
        p.append(1)
      for j in range(args.num_of_repeat): # number of repeat
        cnt = 0 
        sets = p.copy() 
        random.shuffle(sets)
        all_sets.append(sets)
        print(sets)
        print("non overlap count is ",idx)
        file_name = str(idx)+"-"+str(j) 
        fid_score = calculate_fid_given_paths(args.path,
                                          args.batch_size,
                                          True,
                                          args.dims,
                                          sets,name=file_name)
        mse_score, psnr_score, ssim_score, uqi_score, vif_score, brisque_score,vgg_score, lpips_score = test(0,idx,sets, j)
        print(fid_score, mse_score, psnr_score, ssim_score, uqi_score, vif_score, brisque_score,vgg_score, lpips_score)
        cnts.append(idx), mses.append(mse_score), psnrs.append(psnr_score), ssims.append(ssim_score)
        vggs.append(vgg_score), lpipses.append(lpips_score), fids.append(fid_score)
    import matplotlib.pyplot as plt
    print(len(cnts), len(mses), len(psnrs), len(ssims), len(vggs), len(lpipses), len(fids))
    xs = [cnts, mses, psnrs, ssims, vggs, lpipses, fids]
    ys = [cnts, mses, psnrs, ssims, vggs, lpipses, fids]
    xname = ["Number of Inverted pixel", "MSE score", "PSNR score" , "SSIM score", "VGG score", "LPIPS score", "FID score"]
    yname = ["Number of Inverted pixel", "MSE score", "PSNR score" , "SSIM score", "VGG score", "LPIPS score", "FID score"]
    print(xs)
    print(all_sets)
    for xx in range(len(xname)):
      for yy in range(len(xname)):
        if(xx==yy): continue
        plt.figure()
        plt.xlabel(str(xname[xx]))
        plt.ylabel(str(yname[yy]))
        plt.scatter(xs[xx], ys[yy])
        plt.show()
        plt.savefig("rev2_icpr2020_results/"+str(xname[xx])+"_"+str(yname[yy])+".png")


