import torch
import torch.nn as nn
import torchvision
from torchvision import datasets,transforms
import torch.nn.functional as F
from model import CNN
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
def evaluate(device,batch_size = 16):
    model = CNN().to(device)
    model.load_state_dict(torch.load('cnn_mnist.pth', map_location=device))
    model.eval()

    IMAGE_SIZE = 16
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    transforms.ToTensor()
    trans = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    test_ds = datasets.FashionMNIST(root='./data', train=False, download=False, transform=trans)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    images, labels = next(iter( test_loader))
    images, labels = images.to(device), labels.to(device)
    # evaluate
    with torch.no_grad():
        out = model(images)
        pred = out.argmax(1)
    print(f'Predicted: {pred}, Ground Truth: {labels}')
    return images,pred,labels

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image,predicted,label = evaluate(device)
    #convert image to np array
    im_np = image[0, 0].cpu().numpy()
    # 2. Plot
    plt.imshow(im_np, cmap='gray')
    plt.title(f"Predicted: {predicted}   Ground Truth: {label}")
    plt.axis('off')
    plt.show()

    #plot the first 16 examples
    n = 16
    cols = 4
    row = n//cols
    fig,axes = plt.subplots(row,cols,figsize = (cols*2,row*2))
    for i, ax in enumerate(axes.flatten()):
        img_np = image[i].cpu().squeeze().numpy()  # H×W array
        p = predicted[i].item()
        t = label[i].item()

        ax.imshow(img_np, cmap='gray', interpolation='nearest')
        ax.set_title(f"P:{p}  GT:{t}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

#score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
y_trues = label.cpu().numpy() # true labels from test set
y_pred = predicted.cpu().numpy()  # predicted labels from model

#accuracy
acc = accuracy_score(y_trues,y_pred)
#precision
precision =precision_score(y_trues,y_pred,average = 'weighted')
# recall & F1
recall = recall_score(y_trues,y_pred,average = 'macro')
f1 = f1_score(y_trues,y_pred,average = 'macro')

# Confusion matrix
con_mat = confusion_matrix(y_trues,y_pred)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print("Confusion Matrix:")
print(con_mat )



