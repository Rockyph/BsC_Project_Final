import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Perceiver import Perceiver
import wandb  # Import Weights & Biases

# Define the Perceiver model (reuse the provided Perceiver, CrossAttention, SelfAttention, TransformerBlock, PerceiverBlock classes)

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
embedding_size = 128
latent_size = 128
attention_heads = 4
perceiver_depth = 6
transformer_depth = 4
num_classes = 10
epochs = 20
learning_rate = 0.003

# Transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

def main():
    # Initialize Weights & Biases
    api_key = "14037597d70b3d9a3bfb20066d401edf14065e6d"
    wandb.login(key=api_key)
    wandb.init(project="perceiver-cifar10", config={
        "batch_size": batch_size,
        "embedding_size": embedding_size,
        "latent_size": latent_size,
        "attention_heads": attention_heads,
        "perceiver_depth": perceiver_depth,
        "transformer_depth": transformer_depth,
        "num_classes": num_classes,
        "epochs": epochs,
        "learning_rate": learning_rate
    })

    # CIFAR-10 Dataset
    print(f'Device: {device}')
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
        print(f'entering the training loop, we are on: {device}')
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

            # Accumulate loss
            running_loss += loss.item()
            if i % 100 == 99:    # Log every 100 mini-batches
                average_loss = running_loss / 100
                print(f'Epoch {epoch + 1}, Batch {i + 1}: Loss = {average_loss:.3f}')
                wandb.log({"Loss": average_loss})
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

        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}: Accuracy on test set = {accuracy:.2f}%')
        wandb.log({"Test Accuracy": accuracy, "Epoch": epoch + 1})

    print('Finished Training')
    wandb.finish()

if __name__ == '__main__':
    main()
