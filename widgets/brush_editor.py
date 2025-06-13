import numpy as np
from PySide6.QtCore import QObject, Signal, QPoint
from PySide6.QtGui import QPainter, QPen, QBrush, QColor
import cv2

class BrushEditor(QObject):
    mask_changed = Signal()
    skeleton_changed = Signal()
    
    def __init__(self, image_viewer):
        super().__init__()
        self.image_viewer = image_viewer
        
        # Current data
        self.image = None
        self.mask = None
        self.skeleton = None
        
        # Editing state
        self.current_tool = 'draw'  # 'draw' or 'erase'
        self.brush_size = 5
        self.is_drawing = False
        
        # History for undo/redo
        self.mask_history = []
        self.skeleton_history = []
        self.history_index = -1
        
        # Connect to image viewer mouse events
        self.image_viewer.mouse_pressed.connect(self.start_drawing)
        self.image_viewer.mouse_moved.connect(self.draw)
        self.image_viewer.mouse_released.connect(self.stop_drawing)
        
        print("BrushEditor: Initialized and connected to image viewer")
        
    def set_image(self, image):
        """Set the working image"""
        self.image = image.copy() if image is not None else None
        print(f"BrushEditor: Set image - shape: {image.shape if image is not None else None}")
        
    def set_mask(self, mask):
        """Set the current mask"""
        if mask is not None:
            self.mask = mask.copy()
            print(f"BrushEditor: Set mask - shape: {self.mask.shape}, range: [{self.mask.min()}, {self.mask.max()}]")
            # Convert to binary if needed
            if self.mask.dtype != np.uint8:
                self.mask = (self.mask > 0.5).astype(np.uint8)
    
        # Ensure mask dimensions match image dimensions if we have an image
            if self.image is not None:
                image_height, image_width = self.image.shape[:2]
                mask_height, mask_width = self.mask.shape[:2]
                
                if (mask_height, mask_width) != (image_height, image_width):
                    print(f"BrushEditor: Resizing mask from {(mask_width, mask_height)} to match image {(image_width, image_height)}")
                    from skimage.transform import resize
                    self.mask = resize(self.mask, (image_height, image_width), preserve_range=True).astype(np.uint8)
                    
            self.save_mask_state()
        else:
            self.mask = None
            print("BrushEditor: Cleared mask")
                
    def set_skeleton(self, skeleton):
        """Set the current skeleton"""
        if skeleton is not None:
            self.skeleton = skeleton.copy()
            print(f"BrushEditor: Set skeleton - shape: {self.skeleton.shape}")
            
            # Ensure skeleton dimensions match image dimensions if we have an image
            if self.image is not None:
                image_height, image_width = self.image.shape[:2]
                skel_height, skel_width = self.skeleton.shape[:2]
                
                if (skel_height, skel_width) != (image_height, image_width):
                    print(f"BrushEditor: Resizing skeleton from {(skel_width, skel_height)} to match image {(image_width, image_height)}")
                    from skimage.transform import resize
                    self.skeleton = resize(self.skeleton, (image_height, image_width), preserve_range=True).astype(np.uint8)
                    
            self.save_skeleton_state()
        else:
            self.skeleton = None
            print("BrushEditor: Cleared skeleton")
            
    def set_tool(self, tool):
        """Set the current tool ('draw' or 'erase')"""
        self.current_tool = tool
        print(f"BrushEditor: Set tool to {tool}")
        
    def set_brush_size(self, size):
        """Set brush size"""
        self.brush_size = max(1, size)
        print(f"BrushEditor: Set brush size to {self.brush_size}")
        
    def start_drawing(self, pos):
        """Start drawing operation"""
        print(f"BrushEditor: Start drawing at {pos.x()}, {pos.y()}")
        
        if self.image is None:
            print("BrushEditor: No image set, cannot draw")
            return
            
        self.is_drawing = True
        self.apply_brush(pos)
        
    def draw(self, pos):
        """Continue drawing if mouse is pressed"""
        if self.is_drawing and self.image is not None:
            self.apply_brush(pos)
            
    def stop_drawing(self, pos):
        """Stop drawing operation"""
        if self.is_drawing:
            print(f"BrushEditor: Stop drawing at {pos.x()}, {pos.y()}")
            self.is_drawing = False
            # Save state for undo
            if self.mask is not None:
                self.save_mask_state()
            if self.skeleton is not None:
                self.save_skeleton_state()
            
    def apply_brush(self, pos):
        """Apply brush at the given position"""
        if self.image is None:
            print("BrushEditor: No image, cannot apply brush")
            return
            
        x, y = pos.x(), pos.y()
    
        # Get the actual dimensions we need to work with
        if self.current_tool == 'draw':
            # For drawing, we need to ensure mask matches image dimensions
            image_height, image_width = self.image.shape[:2]
            
            if self.mask is None:
                self.mask = np.zeros((image_height, image_width), dtype=np.uint8)
                print(f"BrushEditor: Created new mask with image dimensions: {image_width}x{image_height}")
            elif self.mask.shape != (image_height, image_width):
                # Resize mask to match image if needed
                print(f"BrushEditor: Resizing mask from {self.mask.shape} to {(image_height, image_width)}")
                from skimage.transform import resize
                self.mask = resize(self.mask, (image_height, image_width), preserve_range=True).astype(np.uint8)
                
            height, width = image_height, image_width
            
        elif self.current_tool == 'erase':
            # For erasing, use existing mask/skeleton dimensions
            if self.mask is not None:
                height, width = self.mask.shape[:2]
            elif self.skeleton is not None:
                height, width = self.skeleton.shape[:2]
            else:
                print("BrushEditor: No mask or skeleton to erase from")
                return
        else:
            print(f"BrushEditor: Unknown tool: {self.current_tool}")
            return
        
        print(f"BrushEditor: Apply brush at ({x}, {y}), working dimensions: {width}x{height}, tool: {self.current_tool}")
        
        # Check bounds
        if not (0 <= x < width and 0 <= y < height):
            print(f"BrushEditor: Position ({x}, {y}) out of bounds for {width}x{height}")
            return
            
        # Create brush mask with correct dimensions
        brush_mask = self.create_brush_mask(x, y, height, width)
        brush_pixel_count = np.sum(brush_mask)
        print(f"BrushEditor: Created brush mask {brush_mask.shape} with {brush_pixel_count} pixels")
        
        if self.current_tool == 'draw':
            # Drawing on mask
            old_mask_sum = np.sum(self.mask)
            self.mask[brush_mask] = 1
            new_mask_sum = np.sum(self.mask)
            print(f"BrushEditor: Mask pixels changed from {old_mask_sum} to {new_mask_sum}")
            
            self.mask_changed.emit()
            
        elif self.current_tool == 'erase':
            # Erasing from mask and skeleton
            if self.mask is not None:
                # Ensure brush mask matches mask dimensions
                if brush_mask.shape != self.mask.shape[:2]:
                    print(f"BrushEditor: Brush mask shape {brush_mask.shape} doesn't match mask shape {self.mask.shape[:2]}")
                    return
                    
                old_mask_sum = np.sum(self.mask)
                self.mask[brush_mask] = 0
                new_mask_sum = np.sum(self.mask)
                print(f"BrushEditor: Erased from mask: {old_mask_sum} -> {new_mask_sum}")
                self.mask_changed.emit()
                
            if self.skeleton is not None:
                # Ensure brush mask matches skeleton dimensions  
                if brush_mask.shape != self.skeleton.shape[:2]:
                    print(f"BrushEditor: Brush mask shape {brush_mask.shape} doesn't match skeleton shape {self.skeleton.shape[:2]}")
                    return
                    
                old_skel_sum = np.sum(self.skeleton)
                self.skeleton[brush_mask] = 0
                new_skel_sum = np.sum(self.skeleton)
                print(f"BrushEditor: Erased from skeleton: {old_skel_sum} -> {new_skel_sum}")
                self.skeleton_changed.emit()
                
        # Update display
        print("BrushEditor: Updating display...")
        self.image_viewer.set_mask(self.mask)
        self.image_viewer.set_skeleton(self.skeleton)
        
    def create_brush_mask(self, center_x, center_y, height, width):
        """Create a circular brush mask"""
        y_coords, x_coords = np.ogrid[:height, :width]
        mask = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2 <= self.brush_size ** 2
        return mask
        
    def save_mask_state(self):
        """Save current mask state for undo"""
        if self.mask is not None:
            # Remove future history if we're not at the end
            self.mask_history = self.mask_history[:self.history_index + 1]
            self.mask_history.append(self.mask.copy())
            self.history_index = len(self.mask_history) - 1
            
            print(f"BrushEditor: Saved mask state, history length: {len(self.mask_history)}")
            
            # Limit history size
            max_history = 20
            if len(self.mask_history) > max_history:
                self.mask_history = self.mask_history[-max_history:]
                self.history_index = len(self.mask_history) - 1
                
    def save_skeleton_state(self):
        """Save current skeleton state for undo"""
        if self.skeleton is not None:
            self.skeleton_history = self.skeleton_history[:self.history_index + 1]
            self.skeleton_history.append(self.skeleton.copy())
            
    def undo(self):
        """Undo last operation"""
        if self.history_index > 0:
            self.history_index -= 1
            if self.history_index < len(self.mask_history):
                self.mask = self.mask_history[self.history_index].copy()
                self.image_viewer.set_mask(self.mask)
                self.mask_changed.emit()
                print(f"BrushEditor: Undid to history index {self.history_index}")
                
    def redo(self):
        """Redo last undone operation"""
        if self.history_index < len(self.mask_history) - 1:
            self.history_index += 1
            self.mask = self.mask_history[self.history_index].copy()
            self.image_viewer.set_mask(self.mask)
            self.mask_changed.emit()
            print(f"BrushEditor: Redid to history index {self.history_index}")
            
    def clear_edits(self):
        """Clear all edits"""
        if self.mask is not None:
            self.mask.fill(0)
            self.image_viewer.set_mask(self.mask)
            self.mask_changed.emit()
            self.save_mask_state()
            print("BrushEditor: Cleared mask edits")
            
        if self.skeleton is not None:
            self.skeleton.fill(0)
            self.image_viewer.set_skeleton(self.skeleton)
            self.skeleton_changed.emit()
            print("BrushEditor: Cleared skeleton edits")