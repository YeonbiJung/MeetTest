from torch import nn
import timm

class ConvNextV2(nn.Module):
    def __init__(self, n_outputs:int, **kwargs):
        super(ConvNextV2, self).__init__()
        self.model = timm.create_model('convnextv2_nano', pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_outputs)
        )

    def forward(self, x):
        output = self.model(x)
        return output
    
