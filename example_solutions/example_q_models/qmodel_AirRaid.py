import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetworkPixelAtari(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self,input_channels, action_size, seed, is_dueling):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetworkPixelAtari, self).__init__()
        #self.seed = torch.manual_seed(seed)
        self.is_dueling = is_dueling
        
        # first conv block
        self.conv_module = nn.ModuleList([])
        self.conv_module.append(nn.Conv2d(in_channels=input_channels, out_channels=10,kernel_size=(5, 5))) # 88 X 88 X 10
        self.conv_module.append(nn.ReLU())
        self.conv_module.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))) # 44 X 44 X 10
        
        # second conv block
        self.conv_module.append(nn.Conv2d(in_channels=10, out_channels=20,kernel_size=(5, 5))) # 40 X 40 X 20
        self.conv_module.append(nn.ReLU())
        self.conv_module.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))) # 20 X 20 X 20
        
        # third conv block
        self.conv_module.append(nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(5, 5))) # 16 X 16 X 30
        self.conv_module.append(nn.ReLU())
        self.conv_module.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))) # 8 X 8 X 30
        
        # common fc block
        self.fc_module = nn.ModuleList([])
        self.fc_module.append(nn.Dropout(0.2))
        self.fc_module.append(nn.Linear(1920, 128))
        self.fc_module.append(nn.ReLU())
        self.fc_module.append(nn.Dropout(0.2))
        self.fc_module.append(nn.Linear(128, 64))
        self.fc_module.append(nn.ReLU())
        
        # final outputs
        self.state_action_value_layers = nn.ModuleList([nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, action_size)])
        if is_dueling:
            self.state_value_layers = nn.ModuleList([nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)])
        

    def forward(self, state):
        # conv output
        x = state
        for f in self.conv_module:
            x = f(x)
        x = torch.flatten(x, 1)
        
        # common fc block
        for f in self.fc_module:
            x = f(x)
        
        # Implemented Dueling NetWork
        state_action_value = x
        for f in self.state_action_value_layers:
            state_action_value =  f(state_action_value)
            
        if self.is_dueling:
            avg = state_action_value.mean(dim=1)
            state_value = x
            for f in self.state_value_layers:
                state_value =  f(state_value)
            state_value = (state_value - avg[:,None]).repeat(1,state_action_value.shape[1])
            state_action_value += state_value
        return state_action_value 