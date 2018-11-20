from dataset import GTA, CityScapes, BDD
import torch
from tqdm import tqdm
import numpy as np
from utils import colorEncode
from scipy.io import loadmat
from scipy.misc import imsave
from scipy.stats import entropy
import matplotlib.pyplot as plt

def main():
    colors = loadmat('colormap.mat')['colors']
    
    dataset = GTA(root='/home/selfdriving/datasets/GTA_full', is_train=0)
    h_s, w_s = 720, 1312
    
    #dataset = CityScapes('val', root='/home/selfdriving/datasets/cityscapes_full', is_train=0)
    #h_s, w_s = 720, 1440
    
    #dataset = BDD('val',root='/home/selfdriving/datasets/bdd100k', is_train=0)
    #h_s, w_s = 720, 1280
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        drop_last=False)
    
    count = 0
    statistics = np.zeros([19], dtype=np.int)
    mean_image = torch.zeros([19,h_s,w_s]).to(torch.long)
    for batch_data in tqdm(loader):
        (imgs, segs, infos) = batch_data
        count += imgs.size(0)
        for i in range(19):
            statistics[i] += torch.sum(segs == i)
            mean_image[i] += torch.sum(segs == i, dim=0)
    
    pred = mean_image.data.cpu().numpy()/count
    pred_ = np.argmax(pred, axis=0)
    pred_color = colorEncode(pred_, colors).astype(np.uint8)
    
    #print(entropy(np.ones(19)))
    entropies = entropy(pred)
    plt.imshow(entropies, cmap='hot')
    plt.colorbar()
    plt.savefig('./entropies.png')
    plt.clf()
    
    max_vals = np.max(pred, axis=0)
    plt.imshow(max_vals, cmap='hot')
    plt.colorbar()
    plt.savefig('./max_vals.png')
        
    imsave('./mean.png', pred_color)
    
    #for i in range(19):
    #    print(statistics[i])
        
main()