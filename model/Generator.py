import torch 
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, n_c):
        super(ResBlock, self).__init__()
        
        self.block = nn.Sequential(nn.ReflectionPad2d(1),
                                 nn.Conv2d(n_c, n_c, 3),
                                 nn.InstanceNorm2d(n_c),
                                 nn.ReLU(inplace=True),
                                 nn.ReflectionPad2d(1),
                                 nn.Conv2d(n_c, n_c, 3),
                                 nn.InstanceNorm2d(n_c))
    
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, n_c):
        super(Generator, self).__init__()

        self.numRB = n_c

        entry = [nn.ReflectionPad2d(3),
                nn.Conv2d(3, 64, 7),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True)
        ]

        downsample = [nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.ReLU(inplace=True),
        ]

        resBlocks = [ResBlock(256) for _ in range(self.numRB)]
        upsample = [nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True)
            ]
        
        out = [ nn.ReflectionPad2d(3),
                nn.Conv2d(64, 3, 7),
                nn.Tanh()
                ]
        
        self.body = nn.Sequential(*entry, *downsample, *resBlocks, *upsample, *out)
    
    def forward(self, x):
        return self.body(x)
