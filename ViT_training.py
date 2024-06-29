import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Perceiver import TransformerBlock, SelfAttention
from VIT import VisionTransformer
import wandb  # Import Weights & Biases
from torchmetrics import Accuracy, F1Score  # Import torchmetrics
import tqdm 

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 180
image_size = 32
patch_size = 4
embedding_size = 128
attention_heads = 4
ff = 80
depth = 4
nr_classes = 10
epochs = 150
learning_rate = 0.0003

# Transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(8, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])
def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params

def main():
    # Initialize Weights & Biases
    api_key = "14037597d70b3d9a3bfb20066d401edf14065e6d"
    wandb.login(key=api_key)
    wandb.init(project="vision-transformer-Cifar-100", config={
        "batch_size": batch_size,
        "image_size": image_size,
        "patch_size": patch_size,
        "embedding_size": embedding_size,
        "attention_heads": attention_heads,
        "ff": ff,
        "depth": depth,
        "nr_classes": nr_classes,
        "epochs": epochs,
        "learning_rate": learning_rate
    })

    # CIFAR-10 Dataset
    print(f'Device: {device}')
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    trainset_fmnist = torchvision.datasets.FashionMNIST(root='./data_fmnist', train=True, download=True, transform=transform)
    trainloader_fmnist = DataLoader(trainset_fmnist, batch_size=batch_size, shuffle=True, num_workers=2)

    testset_fmnist = torchvision.datasets.FashionMNIST(root='./data_fmnist', train=False, download=True, transform=transform)
    testloader_fmnist = DataLoader(testset_fmnist, batch_size=batch_size, shuffle=False, num_workers=2)

    trainset_cifar_100 = torchvision.datasets.FashionMNIST(root='./data_cifar_100', train=True, download=True, transform=transform)
    trainloader_cifar_100 = DataLoader(trainset_cifar_100, batch_size=batch_size, shuffle=True, num_workers=2)

    testset_cifar_100 = torchvision.datasets.FashionMNIST(root='./data_cifar_100', train=False, download=True, transform=transform)
    testloader_cifar_100 = DataLoader(testset_cifar_100, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize the model, loss function, and optimizer
    model = VisionTransformer(device=device, channels=3, image_size=image_size, batch_size=batch_size,
                              patch_size=patch_size, embedding_size=embedding_size, attention_heads=attention_heads,
                              ff=ff, depth=depth, nr_classes=nr_classes).to(device)
    print(count_trainable_parameters(model))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize metrics
    accuracy_metric = Accuracy(task="multiclass", num_classes=nr_classes).to(device)
    f1_metric = F1Score(task="multiclass", num_classes=nr_classes, average='macro').to(device)

    # Training Loop
    for epoch in  tqdm.trange(epochs):
        print(f'entering the training loop, we are on: {device}')
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader_cifar_100):
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
            _, preds = torch.max(outputs, 1)
            accuracy_metric.update(preds, labels)
            f1_metric.update(preds, labels)

            if i % 100 == 99:    # Log every 100 mini-batches
                average_loss = running_loss / 100
                train_accuracy = accuracy_metric.compute().item()
                train_f1 = f1_metric.compute().item()
                print(f'Epoch {epoch + 1}, Batch {i + 1}: Loss = {average_loss:.3f}, Accuracy = {train_accuracy:.2f}, F1 Score = {train_f1:.2f}')
                wandb.log({"Loss": average_loss, "Train Accuracy": train_accuracy, "Train F1 Score": train_f1})
                running_loss = 0.0
                accuracy_metric.reset()
                f1_metric.reset()

        # Evaluate the model on the test set
        model.eval()
        correct = 0
        total = 0
        test_accuracy_metric = Accuracy(task="multiclass", num_classes=nr_classes).to(device)
        test_f1_metric = F1Score(task="multiclass", num_classes=nr_classes, average='macro').to(device)
        with torch.no_grad():
            for inputs, labels in testloader_cifar_100:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                test_accuracy_metric.update(predicted, labels)
                test_f1_metric.update(predicted, labels)

        test_accuracy = test_accuracy_metric.compute().item()
        test_f1 = test_f1_metric.compute().item()
        print(f'Epoch {epoch + 1}: Accuracy on test set = {test_accuracy:.2f}%, F1 Score on test set = {test_f1:.2f}')
        wandb.log({"Test Accuracy": test_accuracy, "Test F1 Score": test_f1, "Epoch": epoch + 1})

    print('Finished Training')
    wandb.finish()

if __name__ == '__main__':
    main()
