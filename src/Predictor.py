import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import argparse

# Define the ProductClassifier class (same as in predict.py)
class ProductClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ProductClassifier, self).__init__()
        self.backbone = models.resnet50(pretrained=False) # Important: pretrained=False for inference
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def load_category_mapping(mapping_file_path='category_mapping.json'):
    """Loads the category to index mapping from a JSON file."""
    with open(mapping_file_path, 'r') as f:
        category_mapping = json.load(f)
    return category_mapping

def preprocess_image(image_path, transform=None):
    """Loads and preprocesses an image for prediction."""
    try:
        image = Image.open(image_path).convert('RGB')
        if transform:
            image = transform(image)
        return image
    except Exception as e:
        messagebox.showerror("Error", f"Could not open or process image: {e}")
        return None

def predict_image(image, model, category_mapping, device):
    """Predicts the category of an image."""
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if image is None:
        return None, None

    image_tensor = val_transform(image)
    image_tensor = image_tensor.unsqueeze(0) # Add batch dimension
    image_tensor = image_tensor.to(device)

    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()

    # Use category_mapping directly to get category name from index
    # Keys in category_mapping.json are saved as strings
    predicted_category = category_mapping.get(str(predicted_class_idx), "Unknown Category") # Handle potential missing key

    return predicted_category, probabilities[0][predicted_class_idx].item()


class ImageClassifierGUI:
    def __init__(self, master, model, category_mapping, device):
        self.master = master
        master.title("Product Image Classifier")

        self.model = model
        self.category_mapping = category_mapping
        self.device = device
        self.image_path = None
        self.tk_image = None # To store displayed image

        self.select_image_button = tk.Button(master, text="Select Image", command=self.load_image)
        self.select_image_button.pack(pady=10)

        self.image_label = tk.Label(master, text="No image selected", image=None)
        self.image_label.pack(pady=5)

        self.prediction_label = tk.Label(master, text="Prediction: ")
        self.prediction_label.pack()

        self.probability_label = tk.Label(master, text="Probability: ")
        self.probability_label.pack()

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            initialdir=".",
            title="Select an Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("all files", "*.*"))
        )

        if self.image_path:
            image = preprocess_image(self.image_path)
            if image:
                self.display_image(image)
                self.predict_and_display(image)
            else:
                self.clear_results()


    def display_image(self, img):
        max_size = 250 # Max width/height for display
        img.thumbnail((max_size, max_size)) # Resize to fit in GUI
        self.tk_image = ImageTk.PhotoImage(img) # Convert PIL Image to Tk PhotoImage
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image # Keep a reference!


    def predict_and_display(self, image):
        predicted_category, probability = predict_image(image, self.model, self.category_mapping, self.device)
        if predicted_category:
            self.prediction_label.config(text=f"Prediction: {predicted_category}")
            self.probability_label.config(text=f"Probability: {probability:.4f}")
        else:
            self.clear_results()
            messagebox.showerror("Prediction Error", "Could not make prediction.")


    def clear_results(self):
        self.prediction_label.config(text="Prediction: ")
        self.probability_label.config(text="Probability: ")
        self.image_label.config(image=None, text="No image selected")
        self.image_label.image = None # Clear image reference
        self.tk_image = None


def main():
    parser = argparse.ArgumentParser(description='Predict product category from an image using GUI.')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to the trained model file')
    parser.add_argument('--mapping_path', type=str, default='category_mapping.json', help='Path to the category mapping file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load category mapping
    category_mapping = load_category_mapping(args.mapping_path)
    num_classes = len(category_mapping)

    # Load the trained model
    model = ProductClassifier(num_classes=num_classes).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])


    root = tk.Tk()
    gui = ImageClassifierGUI(root, model, category_mapping, device)
    root.mainloop()


if __name__ == '__main__':
    main()