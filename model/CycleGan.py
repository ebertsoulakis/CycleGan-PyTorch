import torch
import torch.nn as nn

from Utils import ReplayBuffer
from Generator import Generator
from Discriminator import Discriminator

class CycleGAN(nn.Module):
    def __init__(self, n_c, train, eval, argsDict):
        super(CycleGAN, self).__init__():
        self.n_c = n_c
        self.argsDict = argsDict

        if train == True:
            self.train = True
            self.eval = False
        
        if eval == True:
            self.train = False
            self.eval = True

        self.gen_AB = Generator(self.n_c).to(device)
        self.gen_BA = Generator(self.n_c).to(device)

        self.disc_A = Discriminator().to(device)
        self.disc_B = Discriminator().to(device)

        self.gen_AB.apply(init_weights)
        self.gen_BA.apply(init_weights)

        self.disc_A.apply(init_weights)
        self.disc_B.apply(init_weights)

        self.cycle_loss = torch.nn.L1Loss().to(device)
        self.identity_loss = torch.nn.L1Loss().to(device)
        self.adversial_loss = torch.nn.MSELoss.to(device)

        self.optimizer_G = torch.optim.Adam(itertools.chain(gen_AB.parameters(), gen_BA.parameters()),
                               lr=self.argsDict['lr'], betas=(0.5, 0.999))
        self.optimizer_D_A = torch.optim.Adam(disc_A.parameters(), lr=self.argsDict['lr'], betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(disc_B.parameters(), lr=self.argsDict['lr'], betas=(0.5, 0.999))

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()
    
    def forward(self, r_A, r_B, realL, fakeL):

        #Step 1: Generator Networks
        self.optimizer_G.zero_grad()

        #Identity loss
        identity_A = self.genBA(r_A)
        loss_identity_A = self.identity_loss(identity_A, r_A) * 5.0

        identity_B = self.genAB(r_B)
        loss_identity_B = self.identity_loss(identity_B, r_B) * 5.0

        #GAN loss
        fake_A = self.genBA(r_B)
        fake_out_A = self.discA(fake_A)
        loss_GAN_BA = self.adversial_loss(fake_out_A, real_label)

        fake_B = self.genAB(r_A)
        fake_out_B = self.disc_B(fake_B)
        loss_GAN_AB = self.adversial_loss(fake_out_B, real_label)

        #Cycle loss
        recov_A = =self.gen_BA(fake_B)
        loss_cycle_ABA = self.cycle_loss(recov_A, realA) * 10.0

        recov_B = self.genAB(fake_A)
        loss_cycle_BAB = self.cycle_loss(recov_B, realB) * 10.0

        total_error = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

        total_error.backward()
        optimizer_G.step()

        #Step 2: Update Discriminator A
        optimizer_D_A.zero_grad()

        real_out_A = disc_A(realA)
        real_A_err = self.adversial_loss(real_out_A, realA)

        fake_im_A = self.fake_A_buf.push_and_pop(fake_A)
        fake_out_A = self.disc_A(fake_im_A.detach())
        fake_A_err = self.adversial_loss(fake_out_A, fake_label)

        err_disc_A = (real_A_err + fake_A_err) / 2

        err_disc_A.backward()

        optimizer_D_A.step()

        #Step 3: Discriminator B
        optimizer_D_B.zero_grad()

        real_out_B = disc_A(realB)
        real_B_err = self.adversial_loss(real_out_B, realB)

        fake_im_B = self.fake_B_buf.push_and_pop(fake_B)
        fake_out_B = self.disc_B(fake_im_B.detach())
        fake_B_err = self.adversial_loss(fake_out_B, fake_label)

        err_disc_B = (real_B_err + fake_B_err) / 2

        err_disc_B.backward()

        optimizer_D_B.step()

        lossDict = {
            'disc_loss': err_disc_A + err_disc_B,
            'gen_loss': total_loss,
            'gen_identity_loss': loss_identity_A + loss_identity_B,
            'GAN_loss': loss_GAN_BA + loss_GAN_AB,
            'cycle_loss': loss_cycle_ABA + loss_cycle_BAB
        }
        return lossDict






