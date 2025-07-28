import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def train(model, train_loader, test_loader, criterion, optimizer, n_epochs=5, device='cpu'):
    model.to(device)
    train_losses = []
    val_losses = []
    accuracy_list = []
    cost_list = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        cost_list.append(epoch_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_val).sum().item()
                total += y_val.size(0)

        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        accuracy = correct / total
        accuracy_list.append(accuracy)

        print(f"Epoch [{epoch + 1}/{n_epochs}] | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.4f}")

    return train_losses, val_losses, accuracy_list


def plot_metrics(train_losses, val_losses, accuracy_list):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_list, label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model, criterion, optimizer
    model = CNN()
    train_loader, test_loader = get_dataloaders(batch_size=64)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_losses, val_losses, accuracy_list = train(
        model, train_loader, test_loader, criterion, optimizer,
        n_epochs=10, device=device
    )

    plot_metrics(train_losses, val_losses, accuracy_list)
    #save the model
    torch.save(model.state_dict(), 'cnn_mnist.pth')
