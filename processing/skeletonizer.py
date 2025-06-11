import numpy as np
from PySide6.QtCore import QThread, Signal
from skimage.morphology import skeletonize
from scipy.ndimage import binary_erosion, binary_dilation

class Skeletonizer(QThread):
    finished = Signal(np.ndarray)  # skeleton
    progress = Signal(str)
    
    def __init__(self, mask, method="zhang"):
        super().__init__()
        self.mask = mask
        self.method = method
        
    def run(self):
        try:
            self.progress.emit("Starting skeletonization...")
            
            # Clean the mask first
            cleaned_mask = self.clean_mask(self.mask)
            
            self.progress.emit("Running skeletonization algorithm...")
            
            # Skeletonize
            if self.method == "zhang":
                skeleton = skeletonize(cleaned_mask, method='zhang')
            else:  # lee
                skeleton = skeletonize(cleaned_mask, method='lee')
                
            # Post-process skeleton
            skeleton = self.post_process_skeleton(skeleton)
            
            self.progress.emit("Skeletonization completed")
            self.finished.emit(skeleton.astype(np.uint8))
            
        except Exception as e:
            self.progress.emit(f"Error in skeletonization: {str(e)}")
            # Return empty skeleton on error
            empty_skeleton = np.zeros_like(self.mask, dtype=np.uint8)
            self.finished.emit(empty_skeleton)
            
    def clean_mask(self, mask):
        """Clean mask before skeletonization"""
        # Convert to binary
        binary_mask = mask > 0
        
        # Remove small objects
        from skimage.morphology import remove_small_objects
        cleaned = remove_small_objects(binary_mask, min_size=50)
        
        # Fill small holes
        from skimage.morphology import remove_small_holes
        cleaned = remove_small_holes(cleaned, area_threshold=20)
        
        return cleaned.astype(bool)
        
    def post_process_skeleton(self, skeleton):
        """Post-process skeleton to remove artifacts"""
        # Convert to binary
        skeleton_binary = skeleton > 0
        
        # Remove isolated pixels
        from skimage.morphology import remove_small_objects
        skeleton_clean = remove_small_objects(skeleton_binary, min_size=3)
        
        return skeleton_clean
        
    def find_endpoints(self, skeleton):
        """Find skeleton endpoints"""
        import cv2
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        endpoints = (skeleton == 1) & (neighbor_count == 1)
        return endpoints
        
    def find_junctions(self, skeleton):
        """Find skeleton junctions"""
        import cv2
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        junctions = (skeleton == 1) & (neighbor_count > 2)
        return junctions