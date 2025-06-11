import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetLstmModel(nn.Module):
    def __init__(self,hidden_state=512, num_classes=2, lstm_layers=1):
        super(MobileNetLstmModel,self).__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.feature_extractor = mobilenet.features 
        self.pool = nn.AdaptiveAvgPool2d((1,1,))
        self.lstm = nn.LSTM(input_size=1280,
                            hidden_size = hidden_state,
                            num_layers = lstm_layers,
                            batch_first=True,
                            bidirectional = True                
                            )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_state*2,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,num_classes),
        )

    def forward(self,x):
        B,T,C,H,W = x.shape
        x = x.view(B*T,C,H,W)
        x = self.feature_extractor(x)
        # print(self.feature_extractor)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = x.view(B,T,-1)
        output,(hn,cn) = self.lstm(x)
        out = output[:,-1,:]
        # print(out.shape)
        final_out = self.classifier(out)
        # print(final_out)
        # print(final_out.shape)
        return final_out

# data = torch.randn([4,16,3,224,224])
# model = MobileNetLstmModel()
# print(model(data))