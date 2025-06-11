from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion, label as scipy_label
from skimage.morphology import skeletonize
import cv2
import glob
import os

def find_blob_skeleton(blob_mask):
    """Extract skeleton/centerline from blob mask."""
    cleaned_blob = binary_erosion(blob_mask, iterations=1)
    cleaned_blob = binary_dilation(cleaned_blob, iterations=1)
    skeleton = skeletonize(cleaned_blob)
    return skeleton.astype(np.uint8)

def find_endpoints(skeleton):
    """Find endpoints of the skeleton."""
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    endpoints = (skeleton == 1) & (neighbor_count == 1)
    y_coords, x_coords = np.where(endpoints)
    return list(zip(x_coords, y_coords))

def find_intersection_points(skeleton):
    """Find intersection points (junctions) in the skeleton."""
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    intersections = (skeleton == 1) & (neighbor_count > 2)
    y_coords, x_coords = np.where(intersections)
    return list(zip(x_coords, y_coords))

def dilate_and_reskeletonize(skeleton, dilation_iterations=4):
    """Dilate skeleton to connect nearby segments, then re-skeletonize."""
    print(f"  Dilating skeleton by {dilation_iterations} pixels...")
    
    # Dilate the skeleton to connect nearby segments
    dilated_skeleton = binary_dilation(skeleton, iterations=dilation_iterations)
    
    print(f"  Re-skeletonizing dilated structure...")
    
    # Re-skeletonize the dilated structure
    new_skeleton = skeletonize(dilated_skeleton)
    
    return new_skeleton.astype(np.uint8), dilated_skeleton.astype(np.uint8)

def create_trace_from_blob(blob_mask):
    """Create a trace from blob with dilate-reskeletonize approach."""
    print("Step 1: Generating initial skeleton...")
    initial_skeleton = find_blob_skeleton(blob_mask)
    
    initial_endpoints = find_endpoints(initial_skeleton)
    initial_intersections = find_intersection_points(initial_skeleton)
    print(f"Initial skeleton: {len(initial_endpoints)} endpoints, {len(initial_intersections)} intersections")
    
    # Try different dilation amounts from 2 to 5
    results = []
    cells_touching = False
    valid_results = []  # Results that don't cause cell touching
    
    for dilation_amount in [2, 3, 4, 5]:  # Added 2, stopped at 5
        print(f"\nStep 2: Dilating by {dilation_amount} pixels and re-skeletonize...")
        final_skeleton, dilated_structure = dilate_and_reskeletonize(initial_skeleton, dilation_amount)
        
        final_endpoints = find_endpoints(final_skeleton)
        final_intersections = find_intersection_points(final_skeleton)
        print(f"  After dilation {dilation_amount}: {len(final_endpoints)} endpoints, {len(final_intersections)} intersections")
        
        # Check if intersections appeared (cells are touching)
        new_intersections_detected = len(final_intersections) > len(initial_intersections)
        if new_intersections_detected:
            if not cells_touching:
                print(f"  ⚠️  New intersections detected! Cells are touching at dilation {dilation_amount}")
                cells_touching = True
        else:
            # This is a valid result (no cell touching)
            valid_results.append({
                'dilation': dilation_amount,
                'skeleton': final_skeleton,
                'dilated': dilated_structure,
                'endpoints': final_endpoints,
                'intersections': final_intersections,
                'endpoint_count': len(final_endpoints),
                'intersection_count': len(final_intersections),
                'cells_touching': False
            })
        
        results.append({
            'dilation': dilation_amount,
            'skeleton': final_skeleton,
            'dilated': dilated_structure,
            'endpoints': final_endpoints,
            'intersections': final_intersections,
            'endpoint_count': len(final_endpoints),
            'intersection_count': len(final_intersections),
            'cells_touching': new_intersections_detected
        })
    
    # Choose the best result from valid results (no cell touching)
    if valid_results:
        best_result = min(valid_results, key=lambda x: x['endpoint_count'])
        print(f"\nBest valid result: dilation {best_result['dilation']} with {best_result['endpoint_count']} endpoints (no cell touching)")
    else:
        # If all results cause cell touching, use the one with least intersections
        best_result = min(results, key=lambda x: (x['intersection_count'], x['endpoint_count']))
        print(f"\nAll dilations cause cell touching! Using least problematic: dilation {best_result['dilation']}")
    
    if cells_touching:
        print("ℹ️  Note: Some dilations caused cells to touch - avoided in final selection")
    
    return initial_skeleton, results, cells_touching, best_result

def save_intermediate_results(identifier, initial_skeleton, results, cells_touching):
    """Save intermediate processing results to a folder."""
    # Create intermediate folder if it doesn't exist
    intermediate_folder = "intermediate_results"
    os.makedirs(intermediate_folder, exist_ok=True)
    
    # Save initial skeleton
    initial_path = os.path.join(intermediate_folder, f"{identifier}_00_initial_skeleton.png")
    initial_img = Image.fromarray((initial_skeleton * 255).astype(np.uint8))
    initial_img.save(initial_path)
    
    # Save each dilation result
    for result in results:
        dilation = result['dilation']
        
        # Create subfolder for this identifier
        identifier_folder = os.path.join(intermediate_folder, identifier)
        os.makedirs(identifier_folder, exist_ok=True)
        
        # Save dilated structure
        dilated_path = os.path.join(identifier_folder, f"{dilation:02d}_dilated.png")
        dilated_img = Image.fromarray((result['dilated'] * 255).astype(np.uint8))
        dilated_img.save(dilated_path)
        
        # Save final skeleton
        skeleton_path = os.path.join(identifier_folder, f"{dilation:02d}_skeleton.png")
        skeleton_img = Image.fromarray((result['skeleton'] * 255).astype(np.uint8))
        skeleton_img.save(skeleton_path)
        
        # Save metadata
        metadata_path = os.path.join(identifier_folder, f"{dilation:02d}_info.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Dilation: {dilation} pixels\n")
            f.write(f"Endpoints: {result['endpoint_count']}\n")
            f.write(f"Intersections: {result['intersection_count']}\n")
            f.write(f"Cells touching: {result.get('cells_touching', False)}\n")
            f.write(f"Overall cells touching detected: {cells_touching}\n")
    
    print(f"  💾 Intermediate results saved to '{identifier_folder}/' folder")

def save_final_results(identifier, best_result, trace_image):
    """Save final skeletonized results to skeletized folder."""
    # Create skeletized folder if it doesn't exist
    skeletized_folder = "skeletized"
    os.makedirs(skeletized_folder, exist_ok=True)
    
    # Create subfolder for this identifier
    identifier_folder = os.path.join(skeletized_folder, identifier)
    os.makedirs(identifier_folder, exist_ok=True)
    
    # Save final trace
    trace_path = os.path.join(identifier_folder, "final_trace.png")
    trace_pil = Image.fromarray(trace_image.astype(np.uint8))
    trace_pil.save(trace_path)
    
    # Save final skeleton
    skeleton_path = os.path.join(identifier_folder, "final_skeleton.png")
    skeleton_img = Image.fromarray((best_result['skeleton'] * 255).astype(np.uint8))
    skeleton_img.save(skeleton_path)
    
    # Save final metadata
    metadata_path = os.path.join(identifier_folder, "final_info.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"Selected dilation: {best_result['dilation']} pixels\n")
        f.write(f"Final endpoints: {best_result['endpoint_count']}\n")
        f.write(f"Final intersections: {best_result['intersection_count']}\n")
        f.write(f"Cells touching: {best_result.get('cells_touching', False)}\n")
    
    print(f"  📁 Final results saved to '{identifier_folder}/' folder")

def process_segmentation_file(segmentation_path):
    """Process a single segmentation file to create traces."""
    seg_img = Image.open(segmentation_path).convert('L')
    seg_array = np.array(seg_img)
    
    print(f"\nProcessing: {segmentation_path}")
    print(f"Unique values in segmentation: {np.unique(seg_array)}")
    
    cell_wall_blob = (seg_array == 1).astype(np.uint8)
    if np.sum(cell_wall_blob) == 0:
        print("No cell wall pixels found!")
        return None
    
    initial_skeleton, results, cells_touching, best_result = create_trace_from_blob(cell_wall_blob)
    
    # Get identifier for saving
    identifier = os.path.basename(segmentation_path).replace("simple-segmentation-", "").replace(".png", "")
    
    # Always save intermediate results
    save_intermediate_results(identifier, initial_skeleton, results, cells_touching)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Row 1: Original data
    axes[0, 0].imshow(seg_array, cmap='viridis')
    axes[0, 0].set_title('Original Segmentation')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cell_wall_blob, cmap='gray')
    axes[0, 1].set_title('Cell Wall Blob')
    axes[0, 1].axis('off')
    
    # Initial skeleton with endpoints
    axes[0, 2].imshow(initial_skeleton, cmap='gray')
    initial_endpoints = find_endpoints(initial_skeleton)
    initial_intersections = find_intersection_points(initial_skeleton)
    if initial_endpoints:
        end_y, end_x = zip(*[(p[1], p[0]) for p in initial_endpoints])
        axes[0, 2].scatter(end_x, end_y, c='blue', s=20, marker='s', alpha=0.8)
    if initial_intersections:
        int_y, int_x = zip(*[(p[1], p[0]) for p in initial_intersections])
        axes[0, 2].scatter(int_x, int_y, c='red', s=20, marker='o', alpha=0.8)
    axes[0, 2].set_title(f'Initial Skeleton\n{len(initial_endpoints)} endpoints, {len(initial_intersections)} junctions')
    axes[0, 2].axis('off')
    
    # Comparison chart
    dilations = [r['dilation'] for r in results]
    endpoint_counts = [r['endpoint_count'] for r in results]
    intersection_counts = [r['intersection_count'] for r in results]
    
    # Color bars: green for selected, orange for touching, blue for others
    bar_colors = []
    for r in results:
        if r['dilation'] == best_result['dilation']:
            bar_colors.append('green')  # Selected result
        elif r.get('cells_touching', False):
            bar_colors.append('orange')  # Cells touching
        else:
            bar_colors.append('blue')  # Valid but not selected
    
    axes[0, 3].bar([str(d) for d in dilations], endpoint_counts, alpha=0.7, label='Endpoints', color=bar_colors)
    axes[0, 3].bar([str(d) for d in dilations], intersection_counts, alpha=0.7, label='Intersections', bottom=endpoint_counts, color='red')
    title_suffix = f" (Selected: {best_result['dilation']})"
    axes[0, 3].set_title(f'Endpoint/Junction Counts{title_suffix}')
    axes[0, 3].set_xlabel('Dilation Amount')
    axes[0, 3].set_ylabel('Count')
    axes[0, 3].legend()
    
    # Row 2: Dilated structures
    for i, result in enumerate(results):
        axes[1, i].imshow(result['dilated'], cmap='gray')
        selected_marker = " ✓" if result['dilation'] == best_result['dilation'] else ""
        axes[1, i].set_title(f'Dilated Structure\n(dilation: {result["dilation"]}){selected_marker}')
        axes[1, i].axis('off')
    
    # Row 3: Final skeletons with endpoints marked
    for i, result in enumerate(results):
        axes[2, i].imshow(result['skeleton'], cmap='gray')
        
        # Mark endpoints and intersections
        if result['endpoints']:
            end_y, end_x = zip(*[(p[1], p[0]) for p in result['endpoints']])
            axes[2, i].scatter(end_x, end_y, c='blue', s=15, marker='s', alpha=0.8)
        if result['intersections']:
            int_y, int_x = zip(*[(p[1], p[0]) for p in result['intersections']])
            axes[2, i].scatter(int_x, int_y, c='red', s=15, marker='o', alpha=0.8)
        
        # Highlight if this dilation created new intersections or if it's selected
        touching = result.get('cells_touching', False)
        selected = result['dilation'] == best_result['dilation']
        
        if selected:
            title_color = 'green'
            marker = " ✓"
        elif touching:
            title_color = 'red'
            marker = " ⚠️"
        else:
            title_color = 'black'
            marker = ""
        
        axes[2, i].set_title(f'Final Skeleton (dil: {result["dilation"]}){marker}\n{result["endpoint_count"]} endpoints, {result["intersection_count"]} junctions', color=title_color)
        axes[2, i].axis('off')
    
    # Add overall status to the plot
    status_text = f"SELECTED: Dilation {best_result['dilation']} ({best_result['endpoint_count']} endpoints)"
    if cells_touching:
        status_text += " | Some dilations caused cell touching"
    
    fig.suptitle(f'Dilate-Reskeletonize Analysis: {os.path.basename(segmentation_path)}\n{status_text}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Create final trace image
    trace_image = best_result['skeleton'] * 255
    
    # Save final results
    save_final_results(identifier, best_result, trace_image)
    
    return trace_image

def main():
    """Process all segmentation files."""
    # Ensure input folder exists
    if not os.path.exists("simple_segmentation"):
        print("❌ Input folder 'simple_segmentation/' not found!")
        return
    
    segmentation_pattern = os.path.join("simple_segmentation", "simple-segmentation-*.png")
    segmentation_files = glob.glob(segmentation_pattern)
    if not segmentation_files:
        print("No segmentation files found in simple_segmentation/ folder!")
        return
    
    print(f"Found {len(segmentation_files)} segmentation files to process.")
    print(f"Intermediate results will be saved to 'intermediate_results/<identifier>/' folders.")
    print(f"Final results will be saved to 'skeletized/<identifier>/' folders.\n")
    
    for seg_file in segmentation_files:
        try:
            trace_image = process_segmentation_file(seg_file)
            if trace_image is not None:
                identifier = os.path.basename(seg_file).replace("simple-segmentation-", "").replace(".png", "")
                print(f"✅ Completed processing: {identifier}")
        except Exception as e:
            print(f"❌ Error processing {seg_file}: {e}")
            continue

if __name__ == "__main__":
    main()