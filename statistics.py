from dataset import GTA, CityScapes, BDD
import torch
from tqdm import tqdm
import numpy as np

def main():
    dataset = BDD('train',root='/home/selfdriving/datasets/bdd100k')
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=True)

    statistics = np.zeros([19], dtype=np.int)

    for batch_data in tqdm(loader):
        (imgs, segs, infos) = batch_data
        for i in range(19):
            statistics[i] += torch.sum(segs == i)
    for i in range(19):
        print(statistics[i])

main()