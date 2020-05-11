import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from Dataset import processing_all
from Model import RNN
from tensorboardX import SummaryWriter


device=torch.device('cuda:0')
transform=T.ToTensor()
writer=SummaryWriter('D:/new_vocoder')
path='D:/dev-clean/LibriSpeech/dev-clean'
wav_dataset=processing_all(path)
wav_loader=DataLoader(wav_dataset,batch_size=32)

Wav_net=RNN()
Wav_net.to(device)






