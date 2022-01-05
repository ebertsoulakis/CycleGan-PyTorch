import torch
import torch.nn as nn
import itertools

from utils import ReplayBuffer, init_weights
from Generator import Generator
from Discriminator import Discriminator

class CycleGAN(nn.Module):
    def __init__(self, n_c, train, eval, argsDict, device):
        super(CycleGAN, self).__init__()
        self.n_c = n_c
        self.argsDict = argsDict
        self.device = device

        if train == True:
            self.train = True
            self.eval = False
        
        if eval == True:
            self.train = False
            self.eval = True

        self.gen_AB = Generator(self.n_c).to(self.device)
        self.gen_BA = Generator(self.n_c).to(self.device)

        self.disc_A = Discriminator().to(self.device)
        self.disc_B = Discriminator().to(self.device)

        self.gen_AB.apply(init_weights)
        self.gen_BA.apply(init_weights)

        self.disc_A.apply(init_weights)
        self.disc_B.apply(init_weights)

        self.cycle_loss = torch.nn.L1Loss().to(device)
        self.identity_loss = torch.nn.L1Loss().to(device)
        self.adversial_loss = torch.nn.MSELoss().to(device)

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.gen_AB.parameters(), self.gen_BA.parameters()),
                               lr=self.argsDict['lr'], betas=(0.5, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.disc_A.parameters(), lr=self.argsDict['lr'], betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.disc_B.parameters(), lr=self.argsDict['lr'], betas=(0.5, 0.999))

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()
    
    def forward(self, r_A, r_B, realL, fakeL):

        #Step 1: Generator Networks
        self.optimizer_G.zero_grad()

        #Identity loss
        identity_A = self.gen_BA(r_A)
        loss_identity_A = self.identity_loss(identity_A, r_A) * 5.0

        identity_B = self.gen_AB(r_B)
        loss_identity_B = self.identity_loss(identity_B, r_B) * 5.0

        #GAN loss
        fake_A = self.gen_BA(r_B)
        fake_out_A = self.disc_A(fake_A)
        loss_GAN_BA = self.adversial_loss(fake_out_A, realL)

        fake_B = self.gen_AB(r_A)
        fake_out_B = self.disc_B(fake_B)
        loss_GAN_AB = self.adversial_loss(fake_out_B, realL)

        #Cycle loss
        recov_A = self.gen_BA(fake_B)
        loss_cycle_ABA = self.cycle_loss(recov_A, r_A) * 10.0

        recov_B = self.gen_AB(fake_A)
        loss_cycle_BAB = self.cycle_loss(recov_B, r_B) * 10.0

        total_error = loss_identity_A + loss_identity_B + loss_GAN_AB + loss_GAN_BA + loss_cycle_ABA + loss_cycle_BAB

        total_error.backward()
        self.optimizer_G.step()

        #Step 2: Update Discriminator A
        self.optimizer_D_A.zero_grad()

        real_out_A = self.disc_A(r_A)
        real_A_err = self.adversial_loss(real_out_A, r_A)

        fake_im_A = self.fake_A_buffer.push_and_pop(fake_A)
        fake_out_A = self.disc_A(fake_im_A.detach())
        fake_A_err = self.adversial_loss(fake_out_A, fakeL)

        err_disc_A = (real_A_err + fake_A_err) / 2

        err_disc_A.backward()

        self.optimizer_D_A.step()

        #Step 3: Discriminator B
        self.optimizer_D_B.zero_grad()

        real_out_B = self.disc_A(r_B)
        real_B_err = self.adversial_loss(real_out_B, r_B)

        fake_im_B = self.fake_B_buffer.push_and_pop(fake_B)
        fake_out_B = self.disc_B(fake_im_B.detach())
        fake_B_err = self.adversial_loss(fake_out_B, fakeL)

        err_disc_B = (real_B_err + fake_B_err) / 2

        err_disc_B.backward()

        self.optimizer_D_B.step()

        lossDict = {
            'disc_loss': err_disc_A + err_disc_B,
            'gen_loss': total_error,
            'gen_identity_loss': loss_identity_A + loss_identity_B,
            'GAN_loss': loss_GAN_BA + loss_GAN_AB,
            'cycle_loss': loss_cycle_ABA + loss_cycle_BAB
        }
        return lossDict






