import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Perceiver import Perceiver

# Define the Perceiver model (reuse the provided Perceiver, CrossAttention, SelfAttention, TransformerBlock, PerceiverBlock classes)

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
embedding_size = 64
latent_size = 128
attention_heads = 4
perceiver_depth = 6
transformer_depth = 4
num_classes = 10
epochs = 20
learning_rate = 1e-4

# Transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

def main():
    # CIFAR-10 Dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize the model, loss function, and optimizer
    model = Perceiver(device, channels=3, image_size=32, batch_size=batch_size, embedding_size=embedding_size, 
                      latent_size=latent_size, attention_heads=attention_heads, perceiver_depth=perceiver_depth, 
                      transformer_depth=transformer_depth, nr_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(epochs):
        print('entering the training loop')
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            print('i am working')
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}: Loss = {running_loss / 100:.3f}')
                running_loss = 0.0

        # Evaluate the model on the test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch + 1}: Accuracy on test set = {100 * correct / total:.2f}%')

    print('Finished Training')

if __name__ == '__main__':
    main()