import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from classification.dataset import USVDataset
from classification.model import AlexNetClassifier
from utils.logger import setup_logger, get_device_info, create_progress_bar

logger = setup_logger('trainer', 'logs/training.log')
logger.info(f"Using device: {get_device_info()}")

def train_model(annotations_file, audio_dir, num_epochs=10, batch_size=32, learning_rate=0.001):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = USVDataset(annotations_file, audio_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")
    model = AlexNetClassifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Progress bar for training batches
        pbar = create_progress_bar(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), 'alexnet_usv_classifier.pth')

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")
    
    # Rest of your training code...
    # Add progress bars to any other loops as needed
