import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from PySide6.QtCore import QThread, Signal
from pathlib import Path
import traceback

class UNet(nn.Module):
    """U-Net architecture matching your trained model"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class NeuralNetworkProcessor(QThread):
    finished = Signal(np.ndarray, np.ndarray)  # mask, probability_map
    progress = Signal(str)
    
    def __init__(self, image, model_path, threshold):
        super().__init__()
        self.image = image
        self.model_path = model_path
        self.threshold = threshold
        
    def run(self):
        try:
            print(f"DEBUG: Starting neural network processing...")
            print(f"DEBUG: Model path: {self.model_path}")
            print(f"DEBUG: Image shape: {self.image.shape}")
            print(f"DEBUG: Threshold: {self.threshold}")
            
            self.progress.emit("Loading neural network model...")
            
            # Check if we can load the model
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.progress.emit("Model file found, attempting to load...")
            print(f"DEBUG: Model file exists: {Path(self.model_path).exists()}")
            
            # Try to load model using the same logic as test1.py
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"DEBUG: Using device: {device}")
            self.progress.emit(f"Using device: {device}")
            
            # Load the UNet model (same as in test1.py)
            print(f"DEBUG: Creating UNet model...")
            model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
            
            print(f"DEBUG: Loading state dict...")
            try:
                state_dict = torch.load(self.model_path, map_location=device)
                print(f"DEBUG: State dict loaded, keys: {list(state_dict.keys())[:5]}...")
                model.load_state_dict(state_dict)
                model.eval()
                print(f"DEBUG: Model loaded and set to eval mode")
            except Exception as e:
                print(f"DEBUG: Error loading model: {e}")
                traceback.print_exc()
                raise e
            
            self.progress.emit("Model loaded successfully, preprocessing image...")
            
            # Convert numpy array to PIL Image for preprocessing
            print(f"DEBUG: Converting numpy to PIL...")
            if len(self.image.shape) == 3:
                if self.image.max() > 1:
                    pil_image = Image.fromarray(self.image.astype(np.uint8))
                else:
                    pil_image = Image.fromarray((self.image * 255).astype(np.uint8))
            else:
                if self.image.max() > 1:
                    pil_image = Image.fromarray(self.image.astype(np.uint8)).convert('RGB')
                else:
                    pil_image = Image.fromarray((self.image * 255).astype(np.uint8)).convert('RGB')
            
            print(f"DEBUG: PIL image size: {pil_image.size}")
            
            # Preprocess using the same function as test1.py
            print(f"DEBUG: Preprocessing image...")
            input_tensor, original_image, square_image, processed_image = load_and_preprocess_image(pil_image)
            print(f"DEBUG: Input tensor shape: {input_tensor.shape}")
            print(f"DEBUG: Original size: {original_image.size}")
            print(f"DEBUG: Square size: {square_image.size}")
            print(f"DEBUG: Processed size: {processed_image.size}")
            
            self.progress.emit("Running inference...")
            
            # Get prediction (same as test1.py)
            print(f"DEBUG: Running model inference...")
            with torch.no_grad():
                raw_output = model(input_tensor.to(device))
                prediction = torch.sigmoid(raw_output).cpu().squeeze()
            
            print(f"DEBUG: Raw output shape: {raw_output.shape}")
            print(f"DEBUG: Prediction shape: {prediction.shape}")
            print(f"DEBUG: Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")
            
            self.progress.emit("Post-processing prediction...")
            
            # Convert to numpy
            pred_np = prediction.numpy()
            print(f"DEBUG: Prediction numpy shape: {pred_np.shape}")
            
            # Resize prediction back to original image size
            print(f"DEBUG: Resizing prediction back to original size...")
            pred_pil = Image.fromarray((pred_np * 255).astype(np.uint8))
            
            # Resize to square crop size first
            square_size = square_image.size
            pred_square = pred_pil.resize(square_size, Image.LANCZOS)
            print(f"DEBUG: Prediction resized to square: {pred_square.size}")
            
            # Then resize to original size
            original_size = original_image.size
            pred_original = pred_square.resize(original_size, Image.LANCZOS)
            print(f"DEBUG: Prediction resized to original: {pred_original.size}")
            
            # Convert back to numpy array matching input image dimensions
            probability_map = np.array(pred_original) / 255.0
            print(f"DEBUG: Probability map shape: {probability_map.shape}")
            
            # Ensure the probability map matches the input image dimensions
            target_shape = self.image.shape[:2]
            print(f"DEBUG: Target shape: {target_shape}")
            
            if probability_map.shape != target_shape:
                print(f"DEBUG: Resizing from {probability_map.shape} to {target_shape}")
                prob_pil = Image.fromarray((probability_map * 255).astype(np.uint8))
                prob_resized = prob_pil.resize(target_shape[::-1], Image.LANCZOS)
                probability_map = np.array(prob_resized) / 255.0
                print(f"DEBUG: Final probability map shape: {probability_map.shape}")
            
            print(f"DEBUG: Final probability range: [{probability_map.min():.4f}, {probability_map.max():.4f}]")
            self.progress.emit(f"Prediction complete. Prob range: [{probability_map.min():.3f}, {probability_map.max():.3f}]")
            
            # Create binary mask
            mask = (probability_map > self.threshold).astype(np.uint8)
            print(f"DEBUG: Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
            
            # Log some statistics
            coverage = np.sum(mask) / mask.size * 100
            print(f"DEBUG: Coverage: {coverage:.1f}%")
            self.progress.emit(f"Processing completed. Coverage at threshold {self.threshold}: {coverage:.1f}%")
            
            print(f"DEBUG: Emitting results...")
            self.finished.emit(mask, probability_map)
            print(f"DEBUG: Neural network processing completed successfully!")
            
        except Exception as e:
            print(f"DEBUG: Exception in neural network processing: {e}")
            traceback.print_exc()
            self.progress.emit(f"Error: {str(e)}")
            # Emit placeholder data so UI doesn't break
            placeholder_map = self.create_placeholder_mask()
            placeholder_mask = (placeholder_map > self.threshold).astype(np.uint8)
            print(f"DEBUG: Emitting placeholder results...")
            self.finished.emit(placeholder_mask, placeholder_map)
            
    def create_placeholder_mask(self):
        """Create a simple placeholder mask based on intensity"""
        print(f"DEBUG: Creating placeholder mask...")
        self.progress.emit("Creating placeholder mask (no valid model available)")
        
        if len(self.image.shape) == 3:
            gray = np.mean(self.image, axis=2)
        else:
            gray = self.image.copy()
            
        # Normalize to 0-1
        gray = gray.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0
            
        # Simple thresholding as placeholder
        threshold_value = np.mean(gray) + 0.5 * np.std(gray)
        probability_map = np.clip((gray - threshold_value + 0.1) * 5, 0, 1)
        
        print(f"DEBUG: Placeholder probability range: [{probability_map.min():.4f}, {probability_map.max():.4f}]")
        return probability_map

def crop_to_square(image):
    """Crop image to square (center crop) - same as test1.py"""
    w, h = image.size
    min_dim = min(w, h)
    
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    
    return image.crop((left, top, right, bottom))

def load_and_preprocess_image(pil_image, target_size=(384, 384)):
    """Load and preprocess a single image using same pipeline as test1.py"""
    original_image = pil_image.convert('RGB')
    square_image = crop_to_square(original_image)
    resized_image = square_image.resize(target_size, Image.LANCZOS)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(resized_image).unsqueeze(0)
    return tensor, original_image, square_image, resized_image