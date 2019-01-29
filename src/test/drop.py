import torch
import torch.nn as nn

class Model1(nn.Module):
    # Model 1 using functional dropout
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
print('Model 1', model1(inputs))
print('Model 2', model2(inputs))
print()

# switching to eval mode
model1.eval()
model2.eval()

# forwarding inputs in evaluation mode
print('Evaluation mode:')
print('Model 1', model1(inputs))
print('Model 2', model2(inputs))

# show model summary
print('Print summary:')
print(model1)
print(model2)