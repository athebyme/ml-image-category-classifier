import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image, ImageTk, ImageDraw, ImageFont
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import glob


class AdvancedProductClassifier(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b0', pretrained=False, dropout_rate=0.3):
        super(AdvancedProductClassifier, self).__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

        # If using EfficientNet, we can customize the classifier
        if 'efficientnet' in model_name:
            # Get the classifier and replace it with our custom version
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.model(x)


def load_category_mapping(mapping_file_path='category_mapping.json'):
    """Loads the category to index mapping from a JSON file."""
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            category_mapping = json.load(f)
        return category_mapping
    except FileNotFoundError:
        messagebox.showerror("Error", f"Category mapping file not found at: {mapping_file_path}")
        return None
    except json.JSONDecodeError as e:
        messagebox.showerror("Error", f"Error decoding JSON in {mapping_file_path}: {e}")
        return None
    except Exception as e:
        messagebox.showerror("Error", f"Error loading category mapping: {e}")
        return None


def get_transform():
    """Create transform for inference"""
    transform = A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transform


def predict_image(image, model, transform, category_mapping, device, top_k=5):
    """Predicts the top-k categories for an image."""
    model.eval()

    with torch.no_grad():
        # Convert to RGB if grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transformations
        image_np = np.array(image)
        transformed = transform(image=image_np)
        image_tensor = transformed["image"].unsqueeze(0).to(device)

        # Forward pass
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)

    # Convert to Python lists
    top_probs = top_probs.cpu().numpy().tolist()
    top_indices = top_indices.cpu().numpy().tolist()

    # Map indices to category names
    top_categories = [category_mapping.get(str(idx), "Unknown Category") for idx in top_indices]

    return list(zip(top_categories, top_probs))


def apply_class_activation_map(model, image, transform, device):
    """Apply Class Activation Mapping (CAM) to visualize what the models focuses on."""
    # Ensure the models is in eval mode
    model.eval()

    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Original image for overlay
    orig_image = np.array(image)

    # Apply transformations but track the size changes
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    # Apply the transform
    transformed = transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    # Get the features and output
    # This is specific to EfficientNet - adapt for other architectures
    features = None
    gradient = None

    def save_gradient(grad):
        nonlocal gradient
        gradient = grad.detach()

    # Register hooks for the last convolutional layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'conv_head' in name:
            features_hook = module.register_forward_hook(
                lambda module, input, output: setattr(module, 'activations', output)
            )
            module.register_backward_hook(lambda module, grad_in, grad_out: save_gradient(grad_out[0]))
            break

    # Forward pass
    output = model(image_tensor)
    pred_idx = output.argmax(1).item()

    # Zero gradients
    model.zero_grad()

    # Backward pass for the predicted class
    output[0, pred_idx].backward()

    # Get features and gradients
    for name, module in model.named_modules():
        if hasattr(module, 'activations'):
            features = module.activations.detach()
            break

    # Remove hooks
    try:
        features_hook.remove()
    except:
        pass

    # If we couldn't get features or gradients, return original image
    if features is None or gradient is None:
        return orig_image, None

    # Calculate weights and CAM
    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * features, dim=1, keepdim=True)
    cam = torch.relu(cam)  # Apply ReLU to focus on positive contributions

    # Normalize CAM
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-7)  # Add small epsilon to avoid division by zero

    # Resize CAM to original image size
    cam = cam.squeeze().cpu().numpy()
    cam = cv2.resize(cam, (w, h))

    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Convert to RGB (from BGR)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on original image
    alpha = 0.4  # Transparency factor
    overlay = heatmap * alpha + orig_image * (1 - alpha)
    overlay = overlay.astype(np.uint8)

    return overlay, cam


class BatchProcessingFrame(ttk.Frame):
    """Frame for batch processing multiple images."""

    def __init__(self, parent, model, transform, category_mapping, device):
        super().__init__(parent)
        self.parent = parent
        self.model = model
        self.transform = transform
        self.category_mapping = category_mapping
        self.device = device

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.confidence_threshold = tk.DoubleVar(value=0.5)

        self.setup_ui()

    def setup_ui(self):
        # Input directory selection
        ttk.Label(self, text="Input Directory:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(self, textvariable=self.input_dir, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self, text="Browse...", command=self.select_input_dir).grid(row=0, column=2, padx=5, pady=5)

        # Output directory selection
        ttk.Label(self, text="Output Directory:").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(self, textvariable=self.output_dir, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self, text="Browse...", command=self.select_output_dir).grid(row=1, column=2, padx=5, pady=5)

        # Confidence threshold slider
        ttk.Label(self, text="Confidence Threshold:").grid(row=2, column=0, sticky='w', pady=5)
        ttk.Scale(self, variable=self.confidence_threshold, from_=0.0, to=1.0, orient='horizontal').grid(
            row=2, column=1, sticky='ew', padx=5, pady=5)
        ttk.Label(self, textvariable=tk.StringVar(value=lambda: f"{self.confidence_threshold.get():.2f}")).grid(
            row=2, column=2, padx=5, pady=5)

        # Process button
        ttk.Button(self, text="Process Batch", command=self.process_batch).grid(
            row=3, column=0, columnspan=3, pady=10)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=3, sticky='ew', pady=5)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var).grid(
            row=5, column=0, columnspan=3, pady=5)

        # Results text area
        self.results_text = tk.Text(self, height=10, width=60)
        self.results_text.grid(row=6, column=0, columnspan=3, sticky='nsew', pady=5)
        scrollbar = ttk.Scrollbar(self, command=self.results_text.yview)
        scrollbar.grid(row=6, column=3, sticky='ns')
        self.results_text.config(yscrollcommand=scrollbar.set)

        # Make the text area expandable
        self.grid_rowconfigure(6, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def select_input_dir(self):
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_dir.set(directory)

    def select_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)

    def process_batch(self):
        input_dir = self.input_dir.get()
        output_dir = self.output_dir.get()
        threshold = self.confidence_threshold.get()

        if not input_dir or not output_dir:
            messagebox.showerror("Error", "Please select both input and output directories")
            return

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Find all image files
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))

        if not image_files:
            messagebox.showerror("Error", "No image files found in the input directory")
            return

        # Initialize results dictionary
        results = {
            'total': len(image_files),
            'processed': 0,
            'above_threshold': 0,
            'categories': {}
        }

        # Clear results text
        self.results_text.delete(1.0, tk.END)
        self.status_var.set(f"Processing {len(image_files)} images...")

        # Process each image
        for i, image_file in enumerate(image_files):
            try:
                # Update progress
                progress = (i / len(image_files)) * 100
                self.progress_var.set(progress)
                self.update_idletasks()

                # Load and process image
                image = Image.open(image_file).convert('RGB')
                predictions = predict_image(image, self.model, self.transform,
                                            self.category_mapping, self.device)

                # Get top prediction
                top_category, top_prob = predictions[0]

                # Update results
                results['processed'] += 1
                if top_prob >= threshold:
                    results['above_threshold'] += 1

                    # Update category counts
                    if top_category not in results['categories']:
                        results['categories'][top_category] = 0
                    results['categories'][top_category] += 1

                    # Save the image to the appropriate output folder
                    category_dir = os.path.join(output_dir, top_category)
                    os.makedirs(category_dir, exist_ok=True)

                    # Copy image with confidence in filename
                    base_name = os.path.basename(image_file)
                    name, ext = os.path.splitext(base_name)
                    new_name = f"{name}_conf{top_prob:.2f}{ext}"
                    image.save(os.path.join(category_dir, new_name))

                # Update status every 10 images
                if (i + 1) % 10 == 0 or i == len(image_files) - 1:
                    self.status_var.set(f"Processed {i + 1}/{len(image_files)} images")

                    # Update results text
                    self.results_text.delete(1.0, tk.END)
                    self.results_text.insert(tk.END, f"Processed: {results['processed']}/{results['total']}\n")
                    self.results_text.insert(tk.END,
                                             f"Above threshold ({threshold:.2f}): {results['above_threshold']}\n\n")
                    self.results_text.insert(tk.END, "Categories:\n")

                    # Sort categories by count
                    sorted_categories = sorted(results['categories'].items(),
                                               key=lambda x: x[1], reverse=True)

                    for cat, count in sorted_categories:
                        self.results_text.insert(tk.END, f"- {cat}: {count}\n")

                    self.update_idletasks()

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        # Finalize
        self.progress_var.set(100)
        self.status_var.set(f"Completed. Processed {results['processed']} images, "
                            f"{results['above_threshold']} above threshold.")

        # Save summary report
        report_path = os.path.join(output_dir, "processing_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        messagebox.showinfo("Batch Processing Complete",
                            f"Processed {results['processed']} images.\n"
                            f"{results['above_threshold']} were above the confidence threshold.\n"
                            f"Results saved to {output_dir}")


class CamVisualizeFrame(ttk.Frame):
    """Frame for visualizing Class Activation Maps."""

    def __init__(self, parent, model, transform, category_mapping, device):
        super().__init__(parent)
        self.parent = parent
        self.model = model
        self.transform = transform
        self.category_mapping = category_mapping
        self.device = device

        self.image_path = None
        self.current_image = None

        self.setup_ui()

    def setup_ui(self):
        # Control frame
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        ttk.Button(control_frame, text="Select Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Generate CAM", command=self.generate_cam).pack(side=tk.LEFT, padx=5)

        # Images frame
        self.images_frame = ttk.Frame(self)
        self.images_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Original image
        self.orig_frame = ttk.LabelFrame(self.images_frame, text="Original Image")
        self.orig_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        self.orig_label = ttk.Label(self.orig_frame)
        self.orig_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # CAM image
        self.cam_frame = ttk.LabelFrame(self.images_frame, text="Class Activation Map")
        self.cam_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
        self.cam_label = ttk.Label(self.cam_frame)
        self.cam_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Prediction results
        self.results_frame = ttk.LabelFrame(self, text="Prediction Results")
        self.results_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.results_text = tk.Text(self.results_frame, height=5, width=40)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Make the grid cells expandable
        self.images_frame.grid_columnconfigure(0, weight=1)
        self.images_frame.grid_columnconfigure(1, weight=1)
        self.images_frame.grid_rowconfigure(0, weight=1)

    def load_image(self):
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        )

        if not image_path:
            return

        self.image_path = image_path

        try:
            self.current_image = Image.open(image_path).convert('RGB')
            # Resize for display while preserving aspect ratio
            display_image = self.resize_image_for_display(self.current_image, max_size=(400, 400))
            photo = ImageTk.PhotoImage(display_image)

            self.orig_label.config(image=photo)
            self.orig_label.image = photo  # Keep a reference

            # Clear CAM display
            self.cam_label.config(image='')

            # Run prediction
            predictions = predict_image(self.current_image, self.model, self.transform,
                                        self.category_mapping, self.device, top_k=5)

            # Update results
            self.update_prediction_results(predictions)

        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")

    def generate_cam(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first")
            return

        try:
            # Generate CAM
            overlay, _ = apply_class_activation_map(self.model, self.current_image,
                                                    self.transform, self.device)

            # Convert numpy array to PIL Image
            overlay_image = Image.fromarray(overlay)

            # Resize for display
            display_image = self.resize_image_for_display(overlay_image, max_size=(400, 400))
            photo = ImageTk.PhotoImage(display_image)

            # Update display
            self.cam_label.config(image=photo)
            self.cam_label.image = photo  # Keep a reference

        except Exception as e:
            messagebox.showerror("Error", f"Could not generate CAM: {e}")

    def resize_image_for_display(self, image, max_size):
        """Resize image to fit within max_size while preserving aspect ratio."""
        width, height = image.size

        # Calculate ratio
        ratio = min(max_size[0] / width, max_size[1] / height)
        new_size = (int(width * ratio), int(height * ratio))

        return image.resize(new_size, Image.LANCZOS)

    def update_prediction_results(self, predictions):
        """Update the prediction results text area."""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Top predictions:\n")

        for category, prob in predictions:
            self.results_text.insert(tk.END, f"{category}: {prob:.4f}\n")


class ImageClassifierGUI:
    def __init__(self, master, model, category_mapping, device):
        self.master = master
        master.title("Advanced Product Image Classifier")
        master.geometry("900x700")

        self.model = model
        self.category_mapping = category_mapping
        self.device = device
        self.transform = get_transform()

        # Create notebook for tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.single_frame = ttk.Frame(self.notebook)
        self.batch_frame = BatchProcessingFrame(self.notebook, model, self.transform,
                                                category_mapping, device)
        self.cam_frame = CamVisualizeFrame(self.notebook, model, self.transform,
                                           category_mapping, device)

        self.notebook.add(self.single_frame, text="Single Image")
        self.notebook.add(self.batch_frame, text="Batch Process")
        self.notebook.add(self.cam_frame, text="CAM Visualization")

        # Setup single image prediction UI
        self.setup_single_image_ui()

    def setup_single_image_ui(self):
        # Left side - Image display
        left_frame = ttk.Frame(self.single_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Image selection button
        ttk.Button(left_frame, text="Select Image", command=self.load_image).pack(pady=10)

        # Image display
        self.image_frame = ttk.LabelFrame(left_frame, text="Image")
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.image_label = ttk.Label(self.image_frame, text="No image selected")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right side - Prediction results
        right_frame = ttk.Frame(self.single_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

        # Prediction display
        self.prediction_frame = ttk.LabelFrame(right_frame, text="Predictions")
        self.prediction_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create a Figure for the plot
        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Create the canvas for the plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.prediction_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Additional predictions text
        self.prediction_text = tk.Text(right_frame, height=5, width=30)
        self.prediction_text.pack(fill=tk.X, expand=False, pady=5)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        )

        if self.image_path:
            try:
                image = Image.open(self.image_path).convert('RGB')

                # Resize for display
                display_image = self.resize_image_for_display(image)
                self.tk_image = ImageTk.PhotoImage(display_image)

                self.image_label.config(image=self.tk_image, text='')

                # Make prediction
                self.predict_and_display(image)

            except Exception as e:
                messagebox.showerror("Error", f"Could not open or process image: {e}")

    def resize_image_for_display(self, img, max_size=300):
        """Resize image to fit in display area while preserving aspect ratio."""
        width, height = img.size

        if width > height:
            new_width = min(width, max_size)
            new_height = int(height * (new_width / width))
        else:
            new_height = min(height, max_size)
            new_width = int(width * (new_height / height))

        return img.resize((new_width, new_height), Image.LANCZOS)

    def predict_and_display(self, image):
        """Predict categories and display results."""
        try:
            # Get top-k predictions
            predictions = predict_image(image, self.model, self.transform,
                                        self.category_mapping, self.device, top_k=5)

            # Clear the plot
            self.ax.clear()

            # Extract data for plotting
            categories = [p[0] for p in predictions]
            probabilities = [p[1] for p in predictions]

            # Create horizontal bar chart
            bars = self.ax.barh(categories, probabilities, color='skyblue')

            # Add values on the bars
            for bar, prob in zip(bars, probabilities):
                self.ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                             f'{prob:.2f}', va='center')

            # Configure plot
            self.ax.set_title('Category Predictions')
            self.ax.set_xlabel('Probability')
            self.ax.set_xlim(0, 1.1)  # Set x-axis limit with a bit of padding

            # Redraw the plot
            self.fig.tight_layout()
            self.canvas.draw()

            # Update text display
            self.prediction_text.delete(1.0, tk.END)
            self.prediction_text.insert(tk.END, f"Prediction: {predictions[0][0]}\n")
            self.prediction_text.insert(tk.END, f"Probability: {predictions[0][1]:.4f}\n")

            # Add info about the models
            self.prediction_text.insert(tk.END, "\nModel Info:\n")
            self.prediction_text.insert(tk.END, f"Loaded {len(self.category_mapping)} categories")

        except Exception as e:
            self.prediction_text.delete(1.0, tk.END)
            self.prediction_text.insert(tk.END, f"Prediction Error: {e}")
            messagebox.showerror("Prediction Error", f"Could not make prediction: {e}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict product category from an image using GUI.')
    parser.add_argument('--model_path', type=str, default='ensemble_model.pth',
                        help='Path to the trained models file')
    parser.add_argument('--mapping_path', type=str, default='category_mapping.json',
                        help='Path to the category mapping file')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0',
                        help='Model architecture name (from timm)')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load category mapping
    category_mapping = load_category_mapping(args.mapping_path)
    if category_mapping is None:
        return

    num_classes = len(category_mapping)

    # Load the trained models
    try:
        checkpoint = torch.load(args.model_path, map_location=device)

        # Check if this is an ensemble models with additional info
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            # Check if model_name is in the checkpoint
            model_name = checkpoint.get('model_name', args.model_name)
        else:
            # Assume checkpoint is just the state dict
            model_state_dict = checkpoint
            model_name = args.model_name

        # Create models
        model = AdvancedProductClassifier(num_classes=num_classes, model_name=model_name).to(device)
        model.load_state_dict(model_state_dict)
        model.eval()

        print(f"Loaded models: {model_name} with {num_classes} categories")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load models: {e}")
        return

    # Create the GUI
    root = tk.Tk()
    gui = ImageClassifierGUI(root, model, category_mapping, device)
    root.mainloop()


if __name__ == '__main__':
    main()