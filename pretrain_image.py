
import os
import pickle
from gridworld.envs import *
from gridworld.utils.wrapper.wrappers import ImageInputWarpper
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss, MSELoss
from torch import nn
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import numpy as np
import time
import cv2

dataset_path = './dataset/'
image_path = './image/'
if not os.path.exists(image_path):
    os.mkdir(image_path)

# to generate dataset in dataset path
def prepare_dataset():
    def collect_images(env):
        obs = env.reset()
        obs_list=[]
        for i in range(1000):
            obs, reward, done, _=env.step(env.action_space.sample())
            obs_list.append(obs)
            if done:
                env.reset()
        return obs_list

    
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    env = ImageInputWarpper(FourroomsCoinWhiteBackground())
    obs_list = collect_images(env)
    print(len(obs_list))
    with open(dataset_path+"dataset.pkl",'wb') as f:
        pickle.dump(obs_list, f)

class Encoder(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int =256):
        super(Encoder, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = observation_space.shape[0]
        DEPTH = 32
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, DEPTH, kernel_size=4, stride=2, padding=0),
            nn.ELU(),
            nn.Conv2d(DEPTH, DEPTH*2, kernel_size=4, stride=2, padding=0),
            nn.ELU(),
            nn.Conv2d(DEPTH*2, DEPTH*4, kernel_size=4, stride=2, padding=0),
            nn.ELU(),
            # nn.Conv2d(DEPTH*4, DEPTH*8, kernel_size=4, stride=2, padding=0),
            #nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        print(f"n_flatten: {n_flatten}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)#
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_dim, input_channel, shape=(256,2,2)):
        #input:(BATCH, hidden_dim)
        super(Decoder, self).__init__()
        self.shape = shape
        DEPTH = 32
        self.linear = nn.Sequential(nn.Linear(hidden_dim, DEPTH*8*4), nn.ELU())
        self.cnn = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(DEPTH*4, DEPTH*4, kernel_size=3, stride=1, padding=0),
            nn.ELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(DEPTH*4, DEPTH*2, kernel_size=3, stride=1, padding=0),
            nn.ELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(DEPTH*2, DEPTH*2, kernel_size=4, stride=1, padding=0),
            nn.ELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(DEPTH*2, input_channel, kernel_size=3, stride=1, padding=0),
        )
        # self.cnn = nn.Sequential(
        #     nn.ConvTranspose2d(DEPTH*4, DEPTH*4, kernel_size=3, stride=2, padding=0),
        #     nn.ELU(),
        #     nn.ConvTranspose2d(DEPTH*4, DEPTH*4, kernel_size=4, stride=2, padding=0),
        #     nn.ELU(),
        #     nn.ConvTranspose2d(DEPTH*4, DEPTH*2, kernel_size=4, stride=1, padding=0),
        #     nn.ELU(),
        #     nn.ConvTranspose2d(DEPTH*2, DEPTH*2, kernel_size=4, stride=2, padding=0),
        #     nn.ELU(),
        #     nn.ConvTranspose2d(DEPTH*2, input_channel, kernel_size=4, stride=1, padding=0),
        #     #nn.ConvTranspose2d(DEPTH, input_channel, kernel_size=4, stride=2, padding=0),
        # )
        
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # print(features.size())
        batch_size = features.size()[0]
        obs = self.cnn(features)
        obs = torch.tanh(obs)+0.5
        return obs[:,:,:,:]

class ReconNet(nn.Module):
    def __init__(self, encoder:nn.Module):
        super(ReconNet, self).__init__()
        self.encoder = encoder
        self.decoder = Decoder(hidden_dim=512, input_channel=3)

    def forward(self, obs):
        features = nn.ELU()(self.encoder(obs))
        #print(features.size())#(bz,128,6,6)
        return self.decoder(features)


def train_network(recon_network):
    with open(dataset_path+"dataset.pkl", 'rb') as f:
        obs_list = pickle.load(f)
    #create dataloader
    loader = DataLoader(obs_list, batch_size=16, shuffle=True)
    optimizer = Adam(recon_network.parameters(), lr=2e-6)
    print((recon_network))
    loss_f = MSELoss()
    for d in loader:
        data_o = d
        break

    for i in range(50000):
        loss_list = []
        start = time.time()
        printed = False
        #data = next(loader)
        # for data in loader:
        #     if printed == False:
        #         printed = True
        #         sample_image = data[0].numpy().astype(int)

        #         sample_recon = (recon_network(data.permute((0,3,1,2)).float().cuda()/255).permute(0,2,3,1)\
        #         *255)[0].cpu().detach().numpy().astype(int)
        #         cv2.imwrite(image_path+f"origin_{i}.jpg",sample_image)
        #         cv2.imwrite(image_path+f"recon_{i}.jpg",sample_recon)

        #     data = data.permute((0,3,1,2))
        #     data = data.float().cuda()/255
        #     loss = loss_f(data, recon_network(data))/10
        #     loss_list.append(loss.item())
        #     loss.backward()
        #     optimizer.step()
        data = data_o.permute((0,3,1,2))
        data = data.float().cuda()/255
        loss = loss_f(data, recon_network(data))
        loss.backward()
        optimizer.step()
        if i%5000==0:
            print(loss.item())
            end = time.time()
            print(f"epoch {i}")
            # visualize some samples
            sample_image = (data[0]*255).cpu().permute((1,2,0)).numpy().astype(np.uint8)
            sample_recon = (recon_network(data).permute(0,2,3,1)\
                 *255)[0].cpu().detach().numpy().astype(int)
            cv2.imwrite(image_path+f"origin.jpg",sample_image)
            cv2.imwrite(image_path+f"recon_{i}.jpg",sample_recon)
        #print(f"total time: {end - start}")
        #print(np.mean(loss_list))

        
if __name__=="__main__":
    #prepare_dataset()
    print("pid:\t{}".format(os.getpid()))
    encoder = Encoder(gym.spaces.Box(low=0, high=255, shape=(3,64,64), dtype=np.uint8), features_dim=512)
    recon = ReconNet(encoder).cuda()
    train_network(recon)