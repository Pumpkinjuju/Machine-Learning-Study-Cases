import sys
import torch
import os
import matplotlib.pylab as plt
from matplotlib.pyplot import imshow
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import time
from model import *

#import the MNIST data
IMAGE_SIZE = 16
transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
transforms.ToTensor()
trans = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
train_dataset = torchvision.datasets.FashionMNIST(root = "./data", train = True, transform = trans, download = True)
val_dataset = torchvision.datasets.FashionMNIST(root = "./data", train = False, transform = trans, download = True)

def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE,IMAGE_SIZE), cmap = 'gray')
    plt.title('y = ' + str(data_sample[1]))

#print some data of MNIST
for n,data_sample in enumerate(val_dataset):
    show_data(data_sample)
   # plt.show()
    if n == 3:
        break


batch_size =100
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size)
test_loader = DataLoader(dataset = val_dataset, batch_size= batch_size)

#create the model
model = CNN(out_1= 16, out_2 = 32,num_of_class= 10)
print(model)

#train the model
#create the criterion
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

#start training
from train import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_losses, val_losses, accuracy_list = train(model, train_loader, test_loader, criterion, optimizer, n_epochs=10, device=device )
plot_metrics(train_losses, val_losses, accuracy_list)
