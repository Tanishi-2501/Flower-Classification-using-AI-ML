import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Function to load data
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # Define transforms for the training and validation datasets
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    # Using the datasets to define the dataloaders
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    validloader = DataLoader(valid_dataset, batch_size=32, num_workers=4)
    
    return trainloader, validloader

# Function to calculate accuracy
def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Main function that parses arguments and starts training
def main():
    parser = argparse.ArgumentParser(description="Train a neural network model")
    
    parser.add_argument("dataset", type=str, help="Dataset directory for training")
    parser.add_argument("--save_dir", type=str, help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, choices=['vgg16', 'densenet121'], help="Model architecture (vgg16 or densenet121)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units in the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--gpu", action='store_true', help="Use GPU for training if available")
    
    args = parser.parse_args()

    # Debug print to check if the dataset argument is parsed correctly
    print(f"Data directory: {args.dataset}")
    
    # Load the data
    trainloader, validloader = load_data(args.dataset)

    # Example model loading based on architecture choice
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)  # Default model if no architecture provided
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Define a classifier for the model
    model.classifier = nn.Sequential(
        nn.Linear(25088, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(args.hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    # Loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # Move model to GPU if available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()  # Set the model to training mode
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Calculate accuracy on the validation set
        accuracy = calculate_accuracy(model, validloader, device)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {running_loss/len(trainloader):.4f} - Accuracy: {accuracy:.4f}")

    # Save the model checkpoint
    if args.save_dir:
        checkpoint = {
            'state_dict': model.state_dict(),
            'classifier': model.classifier,
            'class_to_idx': trainloader.dataset.class_to_idx
        }
        torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
        print(f"Model saved to {args.save_dir}/checkpoint.pth")

if __name__ == "__main__":
    main()

