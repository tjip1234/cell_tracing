import numpy as np
from PySide6.QtCore import QThread, Signal
from skimage.morphology import skeletonize, binary_dilation, binary_erosion, binary_closing, remove_small_objects, remove_small_holes
from scipy.ndimage import binary_fill_holes, gaussian_filter, distance_transform_edt, label
import cv2

class Skeletonizer(QThread):
    finished = Signal(np.ndarray)  # skeleton
    progress = Signal(str)
    
    def __init__(self, mask, method="zhang", smoothing_level="medium"):
        super().__init__()
        self.mask = mask
        self.method = method
        self.smoothing_level = smoothing_level  # "none", "light", "medium", "heavy"
        
    def run(self):
        try:
            self.progress.emit("Starting skeletonization...")
            
            # Clean the mask first with the new gentle approach
            if self.method in ["gentle", "watershed"]:
                cleaned_mask = self.gentle_mask_cleaning(self.mask)
            else:
                cleaned_mask = self.clean_mask(self.mask)
            
            self.progress.emit("Running skeletonization algorithm...")
            
            # Skeletonize with chosen method
            if self.method == "zhang":
                skeleton = skeletonize(cleaned_mask, method='zhang')
            elif self.method == "lee":
                skeleton = skeletonize(cleaned_mask, method='lee')
            elif self.method == "gentle":
                skeleton = self.gentle_skeletonization(cleaned_mask)
            elif self.method == "watershed":
                skeleton = self.watershed_skeleton(cleaned_mask)
            else:
                skeleton = skeletonize(cleaned_mask, method='zhang')
                
            # Post-process skeleton
            if self.method in ["gentle", "watershed"]:
                skeleton = self.connect_skeleton_fragments(skeleton, cleaned_mask)
            else:
                skeleton = self.post_process_skeleton(skeleton)
            
            self.progress.emit("Skeletonization completed")
            self.finished.emit(skeleton.astype(np.uint8))
            
        except Exception as e:
            self.progress.emit(f"Error in skeletonization: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return empty skeleton on error
            empty_skeleton = np.zeros_like(self.mask, dtype=np.uint8)
            self.finished.emit(empty_skeleton)
            
    def gentle_mask_cleaning(self, mask):
        """Gently clean mask with minimal fragmentation"""
        # Convert to binary
        binary_mask = mask > 0
        
        # Apply smoothing based on level
        if self.smoothing_level == "heavy":
            # Heavy smoothing - good for very noisy masks
            binary_mask = gaussian_filter(binary_mask.astype(float), sigma=2.0) > 0.3
            binary_mask = binary_closing(binary_mask, np.ones((7, 7)))
            binary_mask = remove_small_objects(binary_mask, min_size=100)
            binary_mask = remove_small_holes(binary_mask, area_threshold=50)
            binary_mask = binary_dilation(binary_mask, np.ones((3, 3)))
            
        elif self.smoothing_level == "medium":
            # Medium smoothing - good balance
            binary_mask = gaussian_filter(binary_mask.astype(float), sigma=1.0) > 0.4
            binary_mask = binary_closing(binary_mask, np.ones((5, 5)))
            binary_mask = remove_small_objects(binary_mask, min_size=50)
            binary_mask = remove_small_holes(binary_mask, area_threshold=30)
            binary_mask = binary_dilation(binary_mask, np.ones((2, 2)))
            
        elif self.smoothing_level == "light":
            # Light smoothing - preserves detail
            binary_mask = gaussian_filter(binary_mask.astype(float), sigma=0.5) > 0.5
            binary_mask = binary_closing(binary_mask, np.ones((3, 3)))
            binary_mask = remove_small_objects(binary_mask, min_size=20)
            binary_mask = remove_small_holes(binary_mask, area_threshold=10)
            
        else:  # "none"
            # Minimal cleaning
            binary_mask = remove_small_objects(binary_mask, min_size=10)
            binary_mask = remove_small_holes(binary_mask, area_threshold=5)
        
        return binary_mask.astype(bool)
        
    def gentle_skeletonization(self, mask):
        """Custom gentle skeletonization that's less fragmented"""
        try:
            # Start with distance transform
            distance = distance_transform_edt(mask)
            
            # Simple local maxima finding without skimage.feature
            from scipy.ndimage import maximum_filter
            
            # Find local maxima using maximum filter
            local_maxima = maximum_filter(distance, size=5) == distance
            local_maxima = local_maxima & (distance > 1.0)  # Only consider significant distances
            
            if not np.any(local_maxima):
                # Fallback to regular skeletonization
                return skeletonize(mask)
            
            # Create skeleton from maxima
            skeleton = local_maxima.copy()
            
            # Connect nearby skeleton points with simple dilation
            skeleton = binary_dilation(skeleton, np.ones((3, 3)))
            skeleton = binary_dilation(skeleton, np.ones((3, 3)))  # Second pass for better connectivity
            
            # Thin back to skeleton
            skeleton = skeletonize(skeleton & mask)  # Ensure within mask bounds
            
            return skeleton
            
        except Exception as e:
            print(f"Gentle skeletonization failed: {e}, falling back to zhang")
            return skeletonize(mask, method='zhang')
    
    def watershed_skeleton(self, mask):
        """Use watershed-based approach for smoother skeletons"""
        try:
            # Compute distance transform
            distance = distance_transform_edt(mask)
            
            # Simple local maxima finding
            from scipy.ndimage import maximum_filter
            
            # Find local maxima with larger neighborhood for watershed
            local_maxima = maximum_filter(distance, size=8) == distance
            local_maxima = local_maxima & (distance > 2.0)  # Higher threshold for watershed
            
            if not np.any(local_maxima):
                # Fallback to regular skeletonization
                return skeletonize(mask)
            
            # Label the maxima as markers
            from scipy.ndimage import label as scipy_label
            markers, num_markers = scipy_label(local_maxima)
            
            if num_markers == 0:
                return skeletonize(mask)
            
            # Apply watershed
            from skimage.segmentation import watershed
            labels = watershed(-distance, markers, mask=mask)
            
            # Extract boundaries between regions as skeleton
            skeleton = np.zeros_like(mask, dtype=bool)
            
            # Find region boundaries
            for i in range(1, num_markers + 1):
                region = labels == i
                # Get boundary by subtracting eroded version
                boundary = region & ~binary_erosion(region, np.ones((3, 3)))
                skeleton |= boundary
                
            # Clean up and thin the result
            if np.any(skeleton):
                skeleton = skeleton & mask  # Keep within original mask
                skeleton = skeletonize(skeleton)
            else:
                # If no skeleton found, fallback
                skeleton = skeletonize(mask)
            
            return skeleton
            
        except Exception as e:
            print(f"Watershed skeletonization failed: {e}, falling back to zhang")
            return skeletonize(mask, method='zhang')
        
    def connect_skeleton_fragments(self, skeleton, original_mask):
        """Connect nearby skeleton fragments"""
        # Find skeleton endpoints
        endpoints = self.find_endpoints(skeleton)
        
        # Convert to coordinates
        endpoint_coords = np.where(endpoints)
        if len(endpoint_coords[0]) <= 1:
            return skeleton
            
        endpoints_list = list(zip(endpoint_coords[0], endpoint_coords[1]))
        
        # Connect close endpoints
        connected_skeleton = skeleton.copy()
        
        if len(endpoints_list) > 1:
            # For each endpoint, find nearby endpoints
            for i, (y1, x1) in enumerate(endpoints_list):
                for j, (y2, x2) in enumerate(endpoints_list[i+1:], i+1):
                    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    # Connect if close enough and path is within mask
                    if distance < 15:  # Connection distance
                        if self.can_connect_through_mask(
                            (x1, y1), (x2, y2), original_mask, skeleton
                        ):
                            # Draw line between endpoints
                            line_coords = self.bresenham_line(x1, y1, x2, y2)
                            for lx, ly in line_coords:
                                if (0 <= ly < skeleton.shape[0] and 
                                    0 <= lx < skeleton.shape[1]):
                                    connected_skeleton[ly, lx] = True
                                    
        return connected_skeleton
        
    def can_connect_through_mask(self, point1, point2, mask, skeleton):
        """Check if two points can be connected through the mask"""
        x1, y1 = point1
        x2, y2 = point2
        
        # Get line coordinates
        line_coords = self.bresenham_line(x1, y1, x2, y2)
        
        # Check if most of the line is within the mask
        valid_points = 0
        total_points = len(line_coords)
        
        for x, y in line_coords:
            if (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
                if mask[y, x]:  # Point is within mask
                    valid_points += 1
                elif skeleton[y, x]:  # Point is on existing skeleton
                    valid_points += 1
                    
        # Require at least 70% of points to be valid
        return (valid_points / total_points) > 0.7 if total_points > 0 else False
        
    def bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
                
        return points
            
    def clean_mask(self, mask):
        """Clean mask before skeletonization (original method)"""
        # Convert to binary
        binary_mask = mask > 0
        
        # Remove small objects
        cleaned = remove_small_objects(binary_mask, min_size=50)
        
        # Fill small holes
        cleaned = remove_small_holes(cleaned, area_threshold=20)
        
        return cleaned.astype(bool)
        
    def post_process_skeleton(self, skeleton):
        """Post-process skeleton to remove artifacts (original method)"""
        # Convert to binary
        skeleton_binary = skeleton > 0
        
        # Remove isolated pixels
        skeleton_clean = remove_small_objects(skeleton_binary, min_size=3)
        
        return skeleton_clean
        
    def find_endpoints(self, skeleton):
        """Find skeleton endpoints"""
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        endpoints = (skeleton == 1) & (neighbor_count == 1)
        return endpoints
        
    def find_junctions(self, skeleton):
        """Find skeleton junctions"""
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        junctions = (skeleton == 1) & (neighbor_count > 2)
        return junctions