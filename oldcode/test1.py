import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from train import UNet, device

def crop_to_square(image):
    """Crop image to square (center crop) - same as preprocessing"""
    w, h = image.size
    min_dim = min(w, h)
    
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    
    return image.crop((left, top, right, bottom))

def load_and_preprocess_image(image_path, target_size=(384, 384)):
    """Load and preprocess a single image using same pipeline as training"""
    image = Image.open(image_path).convert('RGB')
    square_image = crop_to_square(image)
    resized_image = square_image.resize(target_size, Image.LANCZOS)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(resized_image).unsqueeze(0)
    return tensor, image, square_image, resized_image

def analyze_single_image(image_path, groundtruth_path=None):
    """Analyze a single image with detailed information and visualizations"""
    
    # Get base filename for outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = f"detailed_analysis_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"="*80)
    print(f"DETAILED ANALYSIS: {base_name}")
    print(f"="*80)
    
    # Load model
    model_paths = ['best_unet_384_model.pth', 'final_unet_384_model.pth']
    model_path = None
    
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print(f"No model file found. Looking for: {model_paths}")
        return
    
    print(f"Loading model from {model_path}...")
    model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded successfully on {device}")
    
    # Load and preprocess image
    print(f"\nLoading and preprocessing image: {image_path}")
    input_tensor, original_image, square_image, processed_image = load_and_preprocess_image(image_path)
    
    print(f"Original size: {original_image.size}")
    print(f"Square crop size: {square_image.size}")
    print(f"Processed size: {processed_image.size}")
    
    # Get prediction with detailed analysis
    print(f"\nRunning inference...")
    with torch.no_grad():
        raw_output = model(input_tensor.to(device))
        prediction = torch.sigmoid(raw_output).cpu().squeeze()
    
    # Convert to numpy for analysis
    raw_np = raw_output.cpu().squeeze().numpy()
    pred_np = prediction.numpy()
    
    print(f"Raw output range: [{raw_np.min():.4f}, {raw_np.max():.4f}]")
    print(f"Sigmoid output range: [{pred_np.min():.4f}, {pred_np.max():.4f}]")
    print(f"Mean probability: {pred_np.mean():.4f}")
    print(f"Std probability: {pred_np.std():.4f}")
    
    # Analyze different thresholds - FIXED: Use consistent thresholds
    thresholds = [0.1, 0.2, 0.3, 0.35, 0.4, 0.48, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_stats = {}
    
    print(f"\nThreshold Analysis:")
    print(f"{'Threshold':<10} {'Coverage %':<12} {'Pixels':<10} {'Connected Components':<20}")
    print("-" * 55)
    
    for thresh in thresholds:
        binary_mask = (prediction > thresh).float().numpy()
        coverage = np.sum(binary_mask > 0.5) / binary_mask.size * 100
        pixel_count = np.sum(binary_mask > 0.5)
        
        # Count connected components (rough estimate)
        try:
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(binary_mask > 0.5)
        except ImportError:
            # Fallback if scipy not available
            num_features = -1
        
        threshold_stats[thresh] = {
            'coverage': coverage,
            'pixels': pixel_count,
            'components': num_features,
            'binary_mask': binary_mask
        }
        
        print(f"{thresh:<10.1f} {coverage:<12.2f} {pixel_count:<10} {num_features:<20}")
    
    # Load ground truth if available
    groundtruth = None
    if groundtruth_path and os.path.exists(groundtruth_path):
        print(f"\nLoading ground truth: {groundtruth_path}")
        gt_img = Image.open(groundtruth_path)
        
        if gt_img.mode == 'RGBA':
            # Convert RGBA to binary mask
            gt_array = np.array(gt_img)
            # Red channel > 0 and alpha > 0 indicates cells
            groundtruth = ((gt_array[:,:,0] > 0) & (gt_array[:,:,3] > 0)).astype(float)
        else:
            gt_gray = gt_img.convert('L')
            gt_array = np.array(gt_gray)
            groundtruth = (gt_array > 128).astype(float)
        
        gt_coverage = np.sum(groundtruth) / groundtruth.size * 100
        print(f"Ground truth coverage: {gt_coverage:.2f}%")
        
        # Calculate metrics for different thresholds
        print(f"\nMetrics vs Ground Truth:")
        print(f"{'Threshold':<10} {'Dice':<8} {'IoU':<8} {'Precision':<10} {'Recall':<8}")
        print("-" * 50)
        
        for thresh in [0.2, 0.35, 0.48, 0.6]:
            if thresh in threshold_stats:  # FIXED: Check if threshold exists
                pred_binary = threshold_stats[thresh]['binary_mask']
                
                # Resize prediction to match ground truth
                if pred_binary.shape != groundtruth.shape:
                    try:
                        from skimage.transform import resize
                        pred_resized = resize(pred_binary, groundtruth.shape, anti_aliasing=False) > 0.5
                    except ImportError:
                        # Fallback: use PIL resize
                        pred_img = Image.fromarray((pred_binary * 255).astype(np.uint8))
                        pred_resized_img = pred_img.resize(groundtruth.shape[::-1], Image.NEAREST)
                        pred_resized = (np.array(pred_resized_img) > 127)
                else:
                    pred_resized = pred_binary > 0.5
                
                # Calculate metrics
                intersection = np.sum(pred_resized & groundtruth)
                union = np.sum(pred_resized | groundtruth)
                pred_sum = np.sum(pred_resized)
                gt_sum = np.sum(groundtruth)
                
                dice = 2 * intersection / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0
                iou = intersection / union if union > 0 else 0
                precision = intersection / pred_sum if pred_sum > 0 else 0
                recall = intersection / gt_sum if gt_sum > 0 else 0
                
                print(f"{thresh:<10.2f} {dice:<8.3f} {iou:<8.3f} {precision:<10.3f} {recall:<8.3f}")
    
    # Create comprehensive visualization
    print(f"\nCreating visualizations...")
    
    # Figure 1: Processing pipeline (no changes needed)
    fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title(f'Original Image\n{original_image.size}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(square_image)
    axes[0, 1].set_title(f'Square Crop\n{square_image.size}')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(processed_image)
    axes[0, 2].set_title(f'Processed (384x384)\nReady for model')
    axes[0, 2].axis('off')
    
    # Raw output
    im_raw = axes[1, 0].imshow(raw_np, cmap='RdBu_r')
    axes[1, 0].set_title(f'Raw Model Output\n[{raw_np.min():.2f}, {raw_np.max():.2f}]')
    axes[1, 0].axis('off')
    plt.colorbar(im_raw, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Probability map
    im_prob = axes[1, 1].imshow(pred_np, cmap='jet', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Probability Map\nMean: {pred_np.mean():.3f}')
    axes[1, 1].axis('off')
    plt.colorbar(im_prob, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Histogram of probabilities
    axes[1, 2].hist(pred_np.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 2].axvline(pred_np.mean(), color='red', linestyle='--', label=f'Mean: {pred_np.mean():.3f}')
    axes[1, 2].axvline(0.2, color='green', linestyle='--', label='Threshold: 0.2')
    axes[1, 2].axvline(0.48, color='orange', linestyle='--', label='Threshold: 0.48')
    axes[1, 2].set_title('Probability Distribution')
    axes[1, 2].set_xlabel('Probability')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_name}_processing_pipeline.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Threshold comparison - FIXED: Use only calculated thresholds
    fig2, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Use only thresholds that were actually calculated
    thresholds_to_show = [t for t in [0.2, 0.35, 0.48, 0.6, 0.7, 0.8] if t in threshold_stats]
    
    for i, thresh in enumerate(thresholds_to_show):
        if i < 8:  # 2x4 grid
            row = i // 4
            col = i % 4
            
            binary_mask = threshold_stats[thresh]['binary_mask']
            coverage = threshold_stats[thresh]['coverage']
            components = threshold_stats[thresh]['components']
            
            # Create overlay
            overlay = np.array(processed_image).copy()
            mask_colored = np.zeros_like(overlay)
            mask_colored[binary_mask > 0.5] = [255, 0, 0]
            alpha = 0.4
            blended = (1 - alpha) * overlay + alpha * mask_colored
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            
            axes[row, col].imshow(blended)
            comp_str = str(components) if components >= 0 else "N/A"
            axes[row, col].set_title(f'Threshold: {thresh}\n{coverage:.1f}% coverage\n{comp_str} components')
            axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(thresholds_to_show), 8):
        row = i // 4
        col = i % 4
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_name}_threshold_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Detailed analysis - Use safe threshold access
    fig3, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Use 0.48 if available, otherwise use 0.5, otherwise use first available threshold
    best_thresh = 0.48 if 0.48 in threshold_stats else (0.5 if 0.5 in threshold_stats else list(threshold_stats.keys())[0])
    
    # Original with overlay
    overlay_orig = np.array(processed_image).copy()
    best_mask = threshold_stats[best_thresh]['binary_mask']
    mask_colored = np.zeros_like(overlay_orig)
    mask_colored[best_mask > 0.5] = [255, 0, 0]
    blended_orig = (0.7 * overlay_orig + 0.3 * mask_colored).astype(np.uint8)
    
    axes[0, 0].imshow(blended_orig)
    axes[0, 0].set_title(f'Prediction Overlay\n(t={best_thresh}, {threshold_stats[best_thresh]["coverage"]:.1f}%)')
    axes[0, 0].axis('off')
    
    # Ground truth comparison if available
    if groundtruth is not None:
        # Resize groundtruth to match prediction
        if groundtruth.shape != best_mask.shape:
            try:
                from skimage.transform import resize
                gt_resized = resize(groundtruth, best_mask.shape, anti_aliasing=False) > 0.5
            except ImportError:
                # Fallback using PIL
                gt_img_temp = Image.fromarray((groundtruth * 255).astype(np.uint8))
                gt_resized_img = gt_img_temp.resize(best_mask.shape[::-1], Image.NEAREST)
                gt_resized = (np.array(gt_resized_img) > 127)
        else:
            gt_resized = groundtruth > 0.5
        
        axes[0, 1].imshow(gt_resized, cmap='gray')
        axes[0, 1].set_title(f'Ground Truth\n{np.sum(gt_resized)/gt_resized.size*100:.1f}% coverage')
        axes[0, 1].axis('off')
        
        # Difference map
        diff_map = np.zeros((*best_mask.shape, 3))
        diff_map[gt_resized & (best_mask > 0.5)] = [0, 1, 0]  # True positive (green)
        diff_map[gt_resized & (best_mask <= 0.5)] = [1, 0, 0]  # False negative (red)
        diff_map[(~gt_resized) & (best_mask > 0.5)] = [0, 0, 1]  # False positive (blue)
        
        axes[0, 2].imshow(diff_map)
        axes[0, 2].set_title('Difference Map\nGreen=TP, Red=FN, Blue=FP')
        axes[0, 2].axis('off')
    else:
        axes[0, 1].text(0.5, 0.5, 'No Ground Truth\nAvailable', ha='center', va='center')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        axes[0, 2].text(0.5, 0.5, 'No Ground Truth\nfor Comparison', ha='center', va='center')
        axes[0, 2].set_title('Difference Map')
        axes[0, 2].axis('off')
    
    # Coverage by threshold
    thresholds_plot = list(threshold_stats.keys())
    coverages_plot = [threshold_stats[t]['coverage'] for t in thresholds_plot]
    
    axes[1, 0].plot(thresholds_plot, coverages_plot, 'bo-', linewidth=2, markersize=6)
    if 0.2 in threshold_stats:
        axes[1, 0].axvline(0.2, color='green', linestyle='--', alpha=0.7, label='t=0.2')
    if 0.48 in threshold_stats:
        axes[1, 0].axvline(0.48, color='orange', linestyle='--', alpha=0.7, label='t=0.48')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Coverage (%)')
    axes[1, 0].set_title('Coverage vs Threshold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Connected components by threshold
    components_plot = [threshold_stats[t]['components'] for t in thresholds_plot]
    
    axes[1, 1].plot(thresholds_plot, components_plot, 'ro-', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Connected Components')
    axes[1, 1].set_title('Components vs Threshold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Statistics text
    best_coverage = threshold_stats[best_thresh]['coverage']
    best_components = threshold_stats[best_thresh]['components']
    alt_coverage = threshold_stats[0.2]['coverage'] if 0.2 in threshold_stats else "N/A"
    alt_components = threshold_stats[0.2]['components'] if 0.2 in threshold_stats else "N/A"
    
    stats_text = f"""
    IMAGE STATISTICS:
    Original size: {original_image.size}
    Processed size: {processed_image.size}
    
    MODEL OUTPUT:
    Raw range: [{raw_np.min():.4f}, {raw_np.max():.4f}]
    Prob range: [{pred_np.min():.4f}, {pred_np.max():.4f}]
    Mean probability: {pred_np.mean():.4f}
    Std probability: {pred_np.std():.4f}
    
    RECOMMENDED THRESHOLD: {best_thresh}
    Coverage at t={best_thresh}: {best_coverage:.2f}%
    Components at t={best_thresh}: {best_components}
    
    ALTERNATIVE THRESHOLD: 0.2
    Coverage at t=0.2: {alt_coverage}%
    Components at t=0.2: {alt_components}
    """
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_name}_detailed_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save detailed statistics to file
    with open(os.path.join(output_dir, f'{base_name}_detailed_stats.txt'), 'w') as f:
        f.write(f"DETAILED ANALYSIS REPORT: {base_name}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"INPUT IMAGE:\n")
        f.write(f"  Path: {image_path}\n")
        f.write(f"  Original size: {original_image.size}\n")
        f.write(f"  Square crop size: {square_image.size}\n")
        f.write(f"  Processed size: {processed_image.size}\n\n")
        
        f.write(f"MODEL OUTPUT:\n")
        f.write(f"  Raw output range: [{raw_np.min():.6f}, {raw_np.max():.6f}]\n")
        f.write(f"  Sigmoid output range: [{pred_np.min():.6f}, {pred_np.max():.6f}]\n")
        f.write(f"  Mean probability: {pred_np.mean():.6f}\n")
        f.write(f"  Std probability: {pred_np.std():.6f}\n")
        f.write(f"  Median probability: {np.median(pred_np):.6f}\n\n")
        
        f.write(f"THRESHOLD ANALYSIS:\n")
        f.write(f"{'Threshold':<10} {'Coverage %':<12} {'Pixels':<10} {'Components':<12}\n")
        f.write("-" * 50 + "\n")
        for thresh, stats in threshold_stats.items():
            f.write(f"{thresh:<10.2f} {stats['coverage']:<12.2f} {stats['pixels']:<10} {stats['components']:<12}\n")
        
        if groundtruth is not None:
            f.write(f"\nGROUND TRUTH COMPARISON:\n")
            f.write(f"  Ground truth path: {groundtruth_path}\n")
            f.write(f"  Ground truth coverage: {np.sum(groundtruth)/groundtruth.size*100:.2f}%\n")
            f.write(f"  Ground truth size: {groundtruth.shape}\n\n")
            
            f.write(f"METRICS (vs Ground Truth):\n")
            f.write(f"{'Threshold':<10} {'Dice':<8} {'IoU':<8} {'Precision':<10} {'Recall':<8}\n")
            f.write("-" * 50 + "\n")
            
            for thresh in [0.2, 0.35, 0.48, 0.6]:
                pred_binary = threshold_stats[thresh]['binary_mask']
                
                if pred_binary.shape != groundtruth.shape:
                    from skimage.transform import resize
                    pred_resized = resize(pred_binary, groundtruth.shape, anti_aliasing=False) > 0.5
                else:
                    pred_resized = pred_binary > 0.5
                
                intersection = np.sum(pred_resized & groundtruth)
                union = np.sum(pred_resized | groundtruth)
                pred_sum = np.sum(pred_resized)
                gt_sum = np.sum(groundtruth)
                
                dice = 2 * intersection / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0
                iou = intersection / union if union > 0 else 0
                precision = intersection / pred_sum if pred_sum > 0 else 0
                recall = intersection / gt_sum if gt_sum > 0 else 0
                
                f.write(f"{thresh:<10.2f} {dice:<8.3f} {iou:<8.3f} {precision:<10.3f} {recall:<8.3f}\n")
    
    # Save binary masks at key thresholds
    for thresh in [0.2, 0.48]:
        binary_mask = threshold_stats[thresh]['binary_mask']
        binary_img = Image.fromarray((binary_mask * 255).astype(np.uint8), mode='L')
        binary_img.save(os.path.join(output_dir, f'{base_name}_binary_t{thresh:.2f}.png'))
    
    # Save probability map as image
    prob_img = Image.fromarray((pred_np * 255).astype(np.uint8), mode='L')
    prob_img.save(os.path.join(output_dir, f'{base_name}_probability_map.png'))
    
    print(f"\nAnalysis completed!")
    print(f"Results saved to: {output_dir}/")
    print(f"Key files:")
    print(f"  - {base_name}_processing_pipeline.png")
    print(f"  - {base_name}_threshold_comparison.png") 
    print(f"  - {base_name}_detailed_analysis.png")
    print(f"  - {base_name}_detailed_stats.txt")
    print(f"  - Binary masks and probability map")

def main():
    """Main function to analyze a single image"""
    
    # Example usage - modify these paths
    image_path = "/home/raaf/plant-cell-segmentation/test.png"
    groundtruth_path = ""  # Optional
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        print("Available images:")
        images_dir = "/home/raaf/plant-cell-segmentation/final-test-set/images"
        if os.path.exists(images_dir):
            for f in os.listdir(images_dir)[:5]:  # Show first 5
                print(f"  {f}")
        return
    
    if groundtruth_path and not os.path.exists(groundtruth_path):
        print(f"Ground truth file not found: {groundtruth_path}")
        print("Proceeding without ground truth comparison...")
        groundtruth_path = None
    
    # Run analysis
    analyze_single_image(image_path, groundtruth_path)

if __name__ == "__main__":
    main()