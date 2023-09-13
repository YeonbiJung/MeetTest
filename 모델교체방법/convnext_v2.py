from torch import nn
import timm

class ConvNextV2(nn.Module):
    def __init__(self, n_outputs:int, **kwargs):
        super(ConvNextV2, self).__init__()
        self.model = timm.create_model('convnextv2_base', pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=1792, out_features=625),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=625, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_outputs)
        )
        print(model)
    def forward(self, x):
        output = self.model(x)
        return output
    
