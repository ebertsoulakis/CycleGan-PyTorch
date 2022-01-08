import argparse
import os
import random
from tqdm import tqdm

from utils import reader, init_weights
from Discriminator import Discriminator
from Generator import Generator
from Dataset import Dataset
from CycleGan import CycleGAN
import torch.backends.cudnn as cudnn
import torch 

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", help="Save Directory for weights")
parser.add_argument("--pretrain", help="Checkpoint to resume training from", default=None)
parser.add_argument("--dataset", help="Path to dataset")
parser.add_argument("--cuda", help="Enable CUDA", action="store_true")
parser.add_argument("--yml", help="Path to yml containing arguments")
parser.add_argument("--train", help="Set model for training", action="store_true")
parser.add_argument("--eval", help="Set model for inference", action="store_true")
args = parser.parse_args()

cudnn.benchmark = True

argsDict = reader(args.yml)

data = Dataset(args.dataset, argsDict['image_size'], True)

dataloader = torch.utils.data.DataLoader(data, batch_size = argsDict['batch_size'], shuffle = True, pin_memory=True)
device = torch.device("cuda:0" if args.cuda else "cpu")

#Build model
model = CycleGAN(9, args.train, args.eval, argsDict, device)
if args.pretrain != None:
    #TODO Implement loading pretrain checkpoints
    CycleGAN.load_state_dict(torch.load(args.pretrain))

#Create save directory
if os.path.exists(args.save_dir) == False:
    os.mkdir(args.save_dir)

os.mkdir(f"{args.save_dir}/weights")


for epoch in range(argsDict['epochs']):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
        realA = data['A'].to(device)
        realB = data['B'].to(device)
        batch_size = realA.size(0)

        real_label = torch.full((batch_size, 1), 1, device=device, dtype=torch.float32)
        fake_label = torch.full((batch_size, 1), 0, device=device, dtype=torch.float32)

        lossDict = model(realA, realB, real_label, fake_label)

        progress_bar.set_description(
            f"[{epoch}/{argsDict['epochs']-1}][{i}/{len(dataloader) - 1}]"
            f" Discriminator Loss: {lossDict['disc_loss'].item():.4f}"
            f" Generator Loss: {lossDict['gen_loss'].item():.4f}"
            f" Identity Loss: {lossDict['gen_identity_loss'].item():.4f}"
            f" GAN Loss: {lossDict['GAN_loss'].item():.4f}"
            f" Cycle Loss: {lossDict['cycle_loss'].item():.4f}"
        )

        #TODO Implement functionality to save images after every epoch

    torch.save(model.state_dict(), f"{args.save_dir}/weights/cyclegan_epoch_{epoch}.pth")
