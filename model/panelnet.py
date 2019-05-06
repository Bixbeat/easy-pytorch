import torch
import torch.nn as nn
import torch.nn.functional as F

class PanelNet(nn.Module):
    def __init__(self, n_classes, in_channels, lyr1_kernels, growth_rate=12):
        super(PanelNetTwo, self).__init__()
        self.n_out = n_classes
        self.in_channels = in_channels
        self.lyr1_kernels = lyr1_kernels
        self.growth_rate = growth_rate
       
        # Downsample path
        self.downblock1 = self.downblock(in_channels, lyr1_kernels, padding=1)
        self.downblock2 = self.downblock(lyr1_kernels, lyr1_kernels+growth_rate, padding=1)
        
        # Pool bottleneck
        self.mpdown1 = nn.MaxPool2d(2, return_indices=True)
        self.downblock3 = self.downblock(lyr1_kernels+growth_rate, lyr1_kernels+(growth_rate*2), padding=1)
        
        self.mpdown2 = nn.MaxPool2d(2, return_indices=True)
        self.downblock4 = self.downblock(lyr1_kernels+(growth_rate*2), lyr1_kernels+(growth_rate*3), padding=1)
        
        self.mpdown3 = nn.MaxPool2d(2, return_indices=True)
        self.midconv1 = nn.Conv2d(in_channels=lyr1_kernels+(growth_rate*3), out_channels=lyr1_kernels+(growth_rate*3), kernel_size=3, stride=1, padding=1)
        self.midconv2 = nn.Conv2d(in_channels=lyr1_kernels+(growth_rate*3), out_channels=lyr1_kernels+(growth_rate*3), kernel_size=1, stride=1, padding=0)
        self.midnorm = nn.BatchNorm2d(lyr1_kernels+(growth_rate*3), momentum=0.2)
        # self.middrop = nn.Dropout2d(p=0.25)        
        self.midrelu = nn.ReLU()

        # Upsample path
        self.mpup1 = nn.MaxUnpool2d(2)
        self.upblock1 = self.upblock((lyr1_kernels+(growth_rate*3))*2, lyr1_kernels+(growth_rate*2), padding=1)        
        
        self.mpup2 = nn.MaxUnpool2d(2)
        self.upblock2 = self.upblock((lyr1_kernels+(growth_rate*2))*2, lyr1_kernels+(growth_rate), padding=1)
        
        self.mpup3 = nn.MaxUnpool2d(2)
        
        self.upblock3 = self.upblock((lyr1_kernels+(growth_rate))*2, lyr1_kernels+(growth_rate), padding=1)
        self.upblock4 = self.upblock((lyr1_kernels*2+growth_rate), lyr1_kernels+growth_rate, padding=1)

        self.finalconv = nn.Conv2d(in_channels=lyr1_kernels+growth_rate, out_channels=self.n_out, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.LogSoftmax(dim=1)

    def downblock(self, in_channels, out_channels, padding=0):
        return nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels, momentum=0.2)
                    # nn.Dropout2d(p=0)
                ])

    def upblock(self, in_channels, out_channels, padding=0):
        return nn.ModuleList([nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels, momentum=0.2)
                    # nn.Dropout2d(p=0)
                ])    

    def forward(self, x):
        for module in self.downblock1:
            x = module(x)
        skip1 = x
        
        for module in self.downblock2:
            x = module(x)
        skip2 = x

        x, indices1 = self.mpdown1(x)                              
        for module in self.downblock3:
            x = module(x)
        skip3 = x 
        
        x, indices2 = self.mpdown2(x)        
        for module in self.downblock4:
            x = module(x)        
        skip4 = x            
        
        x, indices3 = self.mpdown3(x)
        
        x = self.midconv1(x)
        x = self.midconv2(x)
        x = self.midnorm(x)
        # x = self.middrop(x) 
        x = self.midrelu(x)        

        x = self.mpup1(x, indices3)
        
        x = torch.cat([x, skip4], 1)
        for module in self.upblock1:
            x = module(x)        
        
        x = self.mpup2(x, indices2)
        x = torch.cat([x, skip3], 1)
        for module in self.upblock2:
            x = module(x)

        x = self.mpup3(x, indices1)
        
        ### Upsample block ###
        x = torch.cat([x, skip2], 1)
        for module in self.upblock3:
            x = module(x)

        x = torch.cat([x, skip1], 1)
        for module in self.upblock4:
            x = module(x)         

        x = self.finalconv(x)
        x = self.softmax(x)
        return x
