import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal
from scipy.ndimage import binary_dilation, binary_erosion, label as scipy_label
from skimage.morphology import skeletonize
import cv2

class IlastikRefiner(QThread):
    finished = Signal(np.ndarray, dict)  # refined_mask, analysis_results
    progress = Signal(str)
    
    def __init__(self, ilastik_output_path, dilation_range=(2, 5)):
        super().__init__()
        self.ilastik_output_path = ilastik_output_path
        self.dilation_range = dilation_range
        
    def run(self):
        try:
            self.progress.emit("Loading Ilastik segmentation...")
            
            # Load segmentation
            seg_img = Image.open(self.ilastik_output_path).convert('L')
            seg_array = np.array(seg_img)
            
            self.progress.emit(f"Found unique values: {np.unique(seg_array)}")
            
            # Extract cell wall blob (assuming value 1 is cell walls)
            cell_wall_blob = (seg_array == 1).astype(np.uint8)
            
            if np.sum(cell_wall_blob) == 0:
                self.progress.emit("Warning: No cell wall pixels found!")
                self.finished.emit(cell_wall_blob, {})
                return
                
            self.progress.emit("Creating initial skeleton...")
            initial_skeleton = self.find_blob_skeleton(cell_wall_blob)
            
            # Analyze different dilation amounts
            self.progress.emit("Testing different dilation amounts...")
            results = self.analyze_dilations(initial_skeleton)
            
            # Choose best result
            best_result = self.choose_best_result(results)
            
            self.progress.emit(f"Selected dilation {best_result['dilation']} with {best_result['endpoint_count']} endpoints")
            
            analysis_results = {
                'initial_skeleton': initial_skeleton,
                'all_results': results,
                'best_result': best_result,
                'cells_touching': any(r.get('cells_touching', False) for r in results)
            }
            
            self.finished.emit(best_result['skeleton'], analysis_results)
            
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
            
    def find_blob_skeleton(self, blob_mask):
        """Extract skeleton/centerline from blob mask."""
        cleaned_blob = binary_erosion(blob_mask, iterations=1)
        cleaned_blob = binary_dilation(cleaned_blob, iterations=1)
        skeleton = skeletonize(cleaned_blob)
        return skeleton.astype(np.uint8)
        
    def find_endpoints(self, skeleton):
        """Find endpoints of the skeleton."""
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        endpoints = (skeleton == 1) & (neighbor_count == 1)
        y_coords, x_coords = np.where(endpoints)
        return list(zip(x_coords, y_coords))
        
    def find_intersection_points(self, skeleton):
        """Find intersection points (junctions) in the skeleton."""
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        intersections = (skeleton == 1) & (neighbor_count > 2)
        y_coords, x_coords = np.where(intersections)
        return list(zip(x_coords, y_coords))
        
    def dilate_and_reskeletonize(self, skeleton, dilation_iterations):
        """Dilate skeleton to connect nearby segments, then re-skeletonize."""
        dilated_skeleton = binary_dilation(skeleton, iterations=dilation_iterations)
        new_skeleton = skeletonize(dilated_skeleton)
        return new_skeleton.astype(np.uint8), dilated_skeleton.astype(np.uint8)
        
    def analyze_dilations(self, initial_skeleton):
        """Analyze different dilation amounts."""
        initial_endpoints = self.find_endpoints(initial_skeleton)
        initial_intersections = self.find_intersection_points(initial_skeleton)
        
        results = []
        start_dilation, end_dilation = self.dilation_range
        
        for dilation_amount in range(start_dilation, end_dilation + 1):
            self.progress.emit(f"Testing dilation {dilation_amount}...")
            
            final_skeleton, dilated_structure = self.dilate_and_reskeletonize(
                initial_skeleton, dilation_amount
            )
            
            final_endpoints = self.find_endpoints(final_skeleton)
            final_intersections = self.find_intersection_points(final_skeleton)
            
            # Check if intersections appeared (cells are touching)
            new_intersections_detected = len(final_intersections) > len(initial_intersections)
            
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
            
        return results
        
    def choose_best_result(self, results):
        """Choose the best result from analysis."""
        # Prefer results that don't cause cell touching
        valid_results = [r for r in results if not r.get('cells_touching', False)]
        
        if valid_results:
            # Choose the one with minimum endpoints
            best_result = min(valid_results, key=lambda x: x['endpoint_count'])
        else:
            # If all results cause cell touching, choose least problematic
            best_result = min(results, key=lambda x: (x['intersection_count'], x['endpoint_count']))
            
        return best_result