from torch import nn
import torch.nn.functional as F
import torch

class ShipClassifier(nn.Module):
    def __init__(
        self,
        in_channels=1280,
        num_classes=20,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        # classifier
        self.m = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(2, 2)), nn.Flatten(),
            nn.Linear(128*4, 64), 
            nn.ReLU(),
            nn.Linear(64, num_classes)
            )

    def forward(self,input):
        return self.m(input)
    
    def calculate_loss(self,input):
        x = self(input) # model outputs
        x = F.softmax(x) # to probs (20 classes)
        # to a score, many ways to do this!
        x = x * torch.arange(self.num_classes)[None, :].to(x.device)
        return -x.mean()
    
class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)