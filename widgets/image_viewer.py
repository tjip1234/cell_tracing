import numpy as np
from PySide6.QtWidgets import QLabel, QScrollArea, QSizePolicy
from PySide6.QtCore import Qt, Signal, QPoint, QRect
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QWheelEvent, QMouseEvent, QPen

class ImageViewer(QScrollArea):
    mouse_moved = Signal(QPoint)
    mouse_pressed = Signal(QPoint)
    mouse_released = Signal(QPoint)
    
    def __init__(self):
        super().__init__()
        
        # Create image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        self.image_label.setMinimumSize(100, 100)
        
        # Enable mouse tracking
        self.image_label.setMouseTracking(True)
        self.setMouseTracking(True)
        
        # Set as widget for scroll area
        self.setWidget(self.image_label)
        self.setWidgetResizable(False)  # Don't auto-resize the widget
        
        # Image data
        self.original_image = None
        self.mask = None
        self.skeleton = None
        self.scale_factor = 1.0
        
        # Zoom limits
        self.min_scale = 0.1
        self.max_scale = 10.0
        
        # Display options
        self.show_original = True
        self.show_mask = True
        self.show_skeleton = True
        
        # Set placeholder
        self.set_placeholder()
        
        print("ImageViewer: Initialized with zoom capabilities")
        
    def set_placeholder(self):
        """Set placeholder text when no image is loaded"""
        self.image_label.setText("No image loaded")
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: lightgray; color: black;")
        
    def set_image(self, image):
        """Set the original image"""
        print(f"ImageViewer: Setting image with shape {image.shape}, dtype {image.dtype}")
        self.original_image = image.copy()
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: white;")
        self.update_display()
        
    def set_mask(self, mask):
        """Set the mask overlay"""
        if mask is not None:
            print(f"ImageViewer: Setting mask with shape {mask.shape}, dtype {mask.dtype}, range [{mask.min()}, {mask.max()}]")
            # Ensure mask is in proper format
            if mask.dtype == np.float32 or mask.dtype == np.float64:
                # Convert float mask to binary
                self.mask = (mask > 0.5).astype(np.uint8)
            else:
                self.mask = mask.copy()
            print(f"ImageViewer: Processed mask range [{self.mask.min()}, {self.mask.max()}], non-zero pixels: {np.sum(self.mask > 0)}")
        else:
            self.mask = None
        self.update_display()
        
    def set_skeleton(self, skeleton):
        """Set the skeleton overlay"""
        if skeleton is not None:
            print(f"ImageViewer: Setting skeleton with shape {skeleton.shape}, dtype {skeleton.dtype}")
            self.skeleton = skeleton.copy()
        else:
            self.skeleton = None
        self.update_display()
        
    def set_display_options(self, show_original=True, show_mask=True, show_skeleton=True):
        """Set display options"""
        self.show_original = show_original
        self.show_mask = show_mask
        self.show_skeleton = show_skeleton
        print(f"ImageViewer: Display options - Original: {show_original}, Mask: {show_mask}, Skeleton: {show_skeleton}")
        self.update_display()
        
    def update_display(self):
        """Update the displayed image"""
        if self.original_image is None:
            self.set_placeholder()
            return
            
        print("ImageViewer: Updating display...")
        
        try:
            # Ensure we have the right dimensions
            height, width = self.original_image.shape[:2]
            print(f"ImageViewer: Image dimensions: {width}x{height}")
            
            # Convert original image to RGB format
            if len(self.original_image.shape) == 3:
                if self.original_image.shape[2] == 3:
                    # Already RGB
                    rgb_image = self.original_image.copy()
                elif self.original_image.shape[2] == 4:
                    # RGBA, take first 3 channels
                    rgb_image = self.original_image[:, :, :3].copy()
                else:
                    # Unknown format, convert first channel to grayscale
                    gray_img = self.original_image[:, :, 0]
                    rgb_image = np.stack([gray_img, gray_img, gray_img], axis=-1)
            else:
                # Grayscale to RGB
                gray_img = self.original_image
                rgb_image = np.stack([gray_img, gray_img, gray_img], axis=-1)
            
            # Ensure proper data type and range
            if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
                if rgb_image.max() <= 1.0:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                else:
                    rgb_image = rgb_image.astype(np.uint8)
            else:
                rgb_image = rgb_image.astype(np.uint8)
                
            print(f"ImageViewer: RGB image range: [{rgb_image.min()}, {rgb_image.max()}]")
            
            # Start with the base image or create blank
            if self.show_original:
                display_image = rgb_image.copy()
            else:
                display_image = np.zeros((height, width, 3), dtype=np.uint8)
                
            # Add mask overlay
            if self.show_mask and self.mask is not None:
                print("ImageViewer: Adding mask overlay...")
                print(f"ImageViewer: Mask shape: {self.mask.shape}, display image shape: {display_image.shape}")
                
                # Ensure mask is binary
                mask_binary = self.mask > 0
                
                # Resize mask to match image dimensions if needed
                if mask_binary.shape != (height, width):
                    print(f"ImageViewer: Resizing mask from {mask_binary.shape} to {(height, width)}")
                    from PIL import Image
                    mask_pil = Image.fromarray((mask_binary * 255).astype(np.uint8))
                    mask_resized = mask_pil.resize((width, height), Image.NEAREST)
                    mask_binary = np.array(mask_resized) > 127
                    print(f"ImageViewer: Resized mask shape: {mask_binary.shape}")
                
                # Create green overlay with stronger visibility
                mask_indices = mask_binary
                mask_pixel_count = np.sum(mask_indices)
                print(f"ImageViewer: Mask covers {mask_pixel_count} pixels ({mask_pixel_count/(height*width)*100:.1f}%)")
                
                if mask_pixel_count > 0:
                    # Use more visible green with higher alpha
                    alpha = 0.6  # Increased from 0.4
                    green_color = np.array([0, 255, 0], dtype=np.float32)  # Bright green
                    
                    # Apply overlay
                    original_pixels = display_image[mask_indices].astype(np.float32)
                    blended_pixels = (1 - alpha) * original_pixels + alpha * green_color
                    display_image[mask_indices] = blended_pixels.astype(np.uint8)
                    
                    print(f"ImageViewer: Applied green overlay to {mask_pixel_count} pixels")
                else:
                    print("ImageViewer: No mask pixels to overlay")
                
            # Add skeleton overlay
            if self.show_skeleton and self.skeleton is not None:
                print("ImageViewer: Adding skeleton overlay...")
                
                # Ensure skeleton is binary
                if self.skeleton.dtype == np.float32 or self.skeleton.dtype == np.float64:
                    skel_binary = self.skeleton > 0.5
                else:
                    skel_binary = self.skeleton > 0
                
                # Resize skeleton to match image dimensions if needed
                if skel_binary.shape != (height, width):
                    from PIL import Image
                    skel_pil = Image.fromarray((skel_binary * 255).astype(np.uint8))
                    skel_resized = skel_pil.resize((width, height), Image.NEAREST)
                    skel_binary = np.array(skel_resized) > 127
                
                # Add bright red color for skeleton (higher priority than mask)
                skel_indices = skel_binary
                skel_pixel_count = np.sum(skel_indices)
                if skel_pixel_count > 0:
                    display_image[skel_indices] = [255, 0, 0]  # Bright red
                    print(f"ImageViewer: Applied red skeleton overlay to {skel_pixel_count} pixels")
                
            # Convert to QImage
            bytes_per_line = 3 * width
            q_image = QImage(
                display_image.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
            
            if q_image.isNull():
                print("ImageViewer: ERROR - QImage is null!")
                self.image_label.setText("Error: Could not create image")
                return
            
            # Apply scaling
            scaled_size = q_image.size() * self.scale_factor
            scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            print(f"ImageViewer: Created pixmap of size {scaled_pixmap.size().width()}x{scaled_pixmap.size().height()}")
            
            # Set the pixmap
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(scaled_pixmap.size())
            
            # Clear any text
            self.image_label.setText("")
            
            print("ImageViewer: Display update completed successfully")
            
        except Exception as e:
            print(f"ImageViewer: ERROR in update_display: {str(e)}")
            import traceback
            traceback.print_exc()
            self.image_label.setText(f"Error displaying image: {str(e)}")
        
    def wheelEvent(self, event: QWheelEvent):
        """Handle zoom with mouse wheel"""
        if event.modifiers() == Qt.ControlModifier:
            # Zoom
            delta = event.angleDelta().y()
            zoom_factor = 1.15 if delta > 0 else 1/1.15
            
            # Get mouse position for zoom center
            mouse_pos = event.position().toPoint()
            
            # Store old scale and position
            old_scale = self.scale_factor
            
            # Calculate new scale
            new_scale = self.scale_factor * zoom_factor
            new_scale = max(self.min_scale, min(self.max_scale, new_scale))
            
            if new_scale != self.scale_factor:
                self.scale_factor = new_scale
                
                # Zoom towards mouse position
                scroll_x = self.horizontalScrollBar().value()
                scroll_y = self.verticalScrollBar().value()
                
                # Calculate zoom point relative to scroll area
                zoom_x = mouse_pos.x() + scroll_x
                zoom_y = mouse_pos.y() + scroll_y
                
                # Update display
                self.update_display()
                
                # Adjust scroll position to keep zoom point centered
                scale_ratio = new_scale / old_scale
                new_scroll_x = zoom_x * scale_ratio - mouse_pos.x()
                new_scroll_y = zoom_y * scale_ratio - mouse_pos.y()
                
                self.horizontalScrollBar().setValue(int(new_scroll_x))
                self.verticalScrollBar().setValue(int(new_scroll_y))
                
            event.accept()
        else:
            super().wheelEvent(event)
            
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press"""
        if event.button() == Qt.LeftButton:
            # Get position relative to the image label directly
            label_pos = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
            image_pos = self.map_label_to_image_coordinates(label_pos)
            if image_pos is not None:
                print(f"ImageViewer: Mouse press - emitting signal with pos {image_pos.x()}, {image_pos.y()}")
                self.mouse_pressed.emit(image_pos)
            else:
                print("ImageViewer: Mouse press outside image bounds")
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move"""
        label_pos = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
        image_pos = self.map_label_to_image_coordinates(label_pos)
        if image_pos is not None:
            self.mouse_moved.emit(image_pos)
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        if event.button() == Qt.LeftButton:
            label_pos = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
            image_pos = self.map_label_to_image_coordinates(label_pos)
            if image_pos is not None:
                print(f"ImageViewer: Mouse release - emitting signal with pos {image_pos.x()}, {image_pos.y()}")
                self.mouse_released.emit(image_pos)
        super().mouseReleaseEvent(event)
        
    def map_label_to_image_coordinates(self, label_pos):
        """Map label coordinates directly to image coordinates"""
        if self.original_image is None:
            return None
            
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return None
            
        print(f"ImageViewer: Label pos: ({label_pos.x()}, {label_pos.y()})")
        print(f"ImageViewer: Pixmap size: {pixmap.width()}x{pixmap.height()}")
        print(f"ImageViewer: Scale factor: {self.scale_factor}")
        
        # Check if we're within the pixmap bounds
        if not (0 <= label_pos.x() < pixmap.width() and 0 <= label_pos.y() < pixmap.height()):
            print(f"ImageViewer: Position outside pixmap bounds")
            return None
        
        # Account for scaling to get original image coordinates
        image_x = int(label_pos.x() / self.scale_factor)
        image_y = int(label_pos.y() / self.scale_factor)
        
        # Check bounds against original image
        height, width = self.original_image.shape[:2]
        
        print(f"ImageViewer: Calculated image coords: ({image_x}, {image_y})")
        print(f"ImageViewer: Image bounds: {width}x{height}")
        
        if 0 <= image_x < width and 0 <= image_y < height:
            result = QPoint(image_x, image_y)
            print(f"ImageViewer: Valid coordinates: {result.x()}, {result.y()}")
            return result
        else:
            print(f"ImageViewer: Coordinates out of bounds")
            return None
        
    def fit_to_window(self):
        """Fit image to window size"""
        if self.original_image is None:
            return
            
        widget_size = self.size()
        height, width = self.original_image.shape[:2]
        
        scale_x = widget_size.width() / width
        scale_y = widget_size.height() / height
        
        self.scale_factor = min(scale_x, scale_y, 1.0)  # Don't scale up
        self.update_display()
        
    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.scale_factor = 1.0
        self.update_display()
        
    def zoom_in(self):
        """Zoom in by a fixed amount"""
        new_scale = self.scale_factor * 1.25
        new_scale = min(self.max_scale, new_scale)
        if new_scale != self.scale_factor:
            self.scale_factor = new_scale
            self.update_display()
            
    def zoom_out(self):
        """Zoom out by a fixed amount"""
        new_scale = self.scale_factor / 1.25
        new_scale = max(self.min_scale, new_scale)
        if new_scale != self.scale_factor:
            self.scale_factor = new_scale
            self.update_display()
    
    def get_zoom_level(self):
        """Get current zoom level as percentage"""
        return int(self.scale_factor * 100)
        
    def set_zoom_level(self, percentage):
        """Set zoom level by percentage (e.g., 150 for 150%)"""
        new_scale = percentage / 100.0
        new_scale = max(self.min_scale, min(self.max_scale, new_scale))
        if new_scale != self.scale_factor:
            self.scale_factor = new_scale
            self.update_display()
            print(f"ImageViewer: Set zoom to {percentage}%")
    
    def enable_drawing(self, enabled=True):
        """Enable or disable direct drawing on image viewer"""
        # This connects to your existing BrushEditor
        # You might want to add drawing overlay functionality here
        print(f"ImageViewer: Drawing mode {'enabled' if enabled else 'disabled'}")