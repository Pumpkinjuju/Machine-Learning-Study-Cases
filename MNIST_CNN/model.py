 #create a CNN model
import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self, out_1 = 16, out_2 = 32, num_of_class = 10):
        super(CNN,self).__init__()
        self.cnn1 = nn.Conv2d(in_channels = 1, out_channels = out_1, kernel_size = 5, padding = 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        self.cnn2 = nn.Conv2d(in_channels = out_1, out_channels = out_2, kernel_size = 5, stride = 1, padding = 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        # Use a dummy tensor to figure out the flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 16, 16)  # match your actual input
            out = self.maxpool1(F.relu(self.cnn1(dummy)))
            out = self.maxpool2(F.relu(self.cnn2(out)))
            flatten_dim = out.numel()  # now will be 512
        self.fc1 = nn.Linear(flatten_dim, num_of_class)
     #prediction
    def forward(self,x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
      #  print("Flattened size:", x.shape)  # should show [batch, out_2, 7, 7]
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        return x

