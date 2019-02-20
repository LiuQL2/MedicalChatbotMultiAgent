import torch
import torch.nn as nn

class Model1(nn.Module):
    # Model 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN using functional dropout
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, inputs):
        return nn.functional.dropout(inputs, p=self.p, training=True)

class Model2(nn.Module):
    # Model 2 using dropout module
    def __init__(self, p=0.0):
        super().__init__()
        self.drop_layer = torch.nn.Sequential(
            nn.Dropout(p=0)
        )

    def forward(self, inputs):
        return self.drop_layer(inputs)
model1 = Model1(p=0.5) # functional dropout
model2 = Model2(p=0.5) # dropout module

# creating inputs
inputs = torch.rand(10)
# forwarding inputs in train mode
print('Normal (train) model:')
print('Model 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN', model1(inputs))
print('Model 2', model2(inputs))
print()

# switching to eval mode
model1.eval()
model2.eval()

# forwarding inputs in evaluation mode
print('Evaluation mode:')
print('Model 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN', model1(inputs))
print('Model 2', model2(inputs))

# show model summary
print('Print summary:')
print(model1)
print(model2)