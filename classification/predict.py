import torch
from torchvision import transforms
from PIL import Image
from classification.model import AlexNetClassifier
from utils.logger import setup_logger, get_device_info

logger = setup_logger('predictor', 'logs/prediction.log')
logger.info(f"Using device: {get_device_info()}")

def predict(image_path, model_path='alexnet_usv_classifier.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running predictions on {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    model = AlexNetClassifier(num_classes=12).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(image.to(device))
        _, predicted = torch.max(outputs, 1)

    return predicted.item()
