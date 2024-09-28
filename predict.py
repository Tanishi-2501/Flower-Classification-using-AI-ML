import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json

# Argument Parser
def get_input_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    
    parser.add_argument('input_img', type=str, help='Image file path')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file path')
    parser.add_argument('--top_k', type=int, default=5, help='Top K classes')
    parser.add_argument('--category_names', type=str, help='Path to category to name JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    return parser.parse_args()

# Load checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    # Load the pre-trained model
    model = models.vgg16(pretrained=True)
    
    # Rebuild the classifier from the checkpoint
    input_size = 25088  # This is specific to VGG16
    hidden_units = 4096  # Based on the VGG16 architecture
    output_size = 102  # Number of flower classes

    # Ensure the classifier matches the checkpoint's architecture
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_units),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(hidden_units, output_size),  # Match the size from the checkpoint
        torch.nn.LogSoftmax(dim=1)
    )
    
    # Load the model's state_dict from the checkpoint
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load the class_to_idx mapping
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

    
   

# Process image
def process_image(image_path):
    pil_image = Image.open(image_path)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = preprocess(pil_image)
    
    return img_tensor

# Predict classes
def predict(image_path, model, topk=5, device='cpu'):
    model.eval()
    
    img_tensor = process_image(image_path)
    img_tensor = img_tensor.unsqueeze_(0).to(device)
    
    with torch.no_grad():
        logps = model.forward(img_tensor)
    
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    
    top_p = top_p.cpu().numpy().flatten()
    top_class = top_class.cpu().numpy().flatten()
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_class = [idx_to_class[c] for c in top_class]
    
    return top_p, top_class

# Main function for running predictions
def main():
    args = get_input_args()
    
    # Load the model checkpoint
    model = load_checkpoint(args.checkpoint)
    
    # Check for GPU
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Predict
    probs, classes = predict(args.input_img, model, topk=args.top_k, device=device)
    
    # Load class names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[c] for c in classes]
    
    # Print results
    for prob, class_name in zip(probs, classes):
        print(f"{class_name}: {prob*100:.2f}%")

if __name__ == '__main__':
    main()

