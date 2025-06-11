import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QFileDialog, QTabWidget, QGroupBox,
    QSlider, QSpinBox, QDoubleSpinBox, QProgressBar, QTextEdit,
    QComboBox, QCheckBox, QSplitter, QFrame, QButtonGroup,
    QToolButton, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QPixmap, QIcon, QAction, QKeySequence
import numpy as np

from widgets.image_viewer import ImageViewer
from widgets.brush_editor import BrushEditor
from processing.neural_network import NeuralNetworkProcessor
from processing.ilastik_refiner import IlastikRefiner
from processing.skeletonizer import Skeletonizer
from processing.endpoint_connector import EndpointConnector
from processing.svg_exporter import SVGExporter

def crop_image_to_square(image):
    """Crop numpy image to square (center crop) - same logic as neural network preprocessing"""
    if len(image.shape) == 3:
        h, w = image.shape[:2]
    else:
        h, w = image.shape
    
    min_dim = min(w, h)
    
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    
    if len(image.shape) == 3:
        return image[top:bottom, left:right, :]
    else:
        return image[top:bottom, left:right]

class CellTracingMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cell Tracing Widget")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data
        self.original_image = None
        self.display_image = None  # Square cropped version for display
        self.current_mask = None
        self.current_skeleton = None
        self.probability_map = None
        
        # Processing threads
        self.nn_processor = None
        self.ilastik_refiner = None
        self.skeletonizer = None
        self.endpoint_connector = None
        
        # Setup UI
        self.setup_ui()
        self.setup_menus()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the main user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls
        self.setup_control_panel(splitter)
        
        # Right panel - Image viewer and editor
        self.setup_image_panel(splitter)
        
        # Set splitter proportions
        splitter.setSizes([400, 1000])
        
        # Status bar
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
    def setup_control_panel(self, parent):
        """Setup the left control panel"""
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.StyledPanel)
        control_frame.setMaximumWidth(450)
        parent.addWidget(control_frame)
        
        layout = QVBoxLayout(control_frame)
        
        # Tab widget for different workflows
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # UNet tab
        self.setup_unet_tab()
        
        # Ilastik tab
        self.setup_ilastik_tab()
        
        # Manual editing tab
        self.setup_editing_tab()
        
        # Export tab
        self.setup_export_tab()
        
        # Log window
        self.setup_log_window(layout)
        
    def setup_unet_tab(self):
        """Setup UNet processing tab"""
        unet_widget = QWidget()
        layout = QVBoxLayout(unet_widget)
        
        # Image selection
        image_group = QGroupBox("Input Image")
        image_layout = QVBoxLayout(image_group)
        
        self.image_path_label = QLabel("No image selected")
        self.image_path_label.setWordWrap(True)
        image_layout.addWidget(self.image_path_label)
        
        self.select_image_btn = QPushButton("Select Image")
        self.select_image_btn.clicked.connect(self.select_image)
        image_layout.addWidget(self.select_image_btn)
        
        layout.addWidget(image_group)
        
        # Model settings
        model_group = QGroupBox("UNet Model")
        model_layout = QVBoxLayout(model_group)
        
        self.model_path_label = QLabel("No model selected")
        self.model_path_label.setWordWrap(True)
        model_layout.addWidget(self.model_path_label)
        
        self.select_model_btn = QPushButton("Select Model (.pth)")
        self.select_model_btn.clicked.connect(self.select_model)
        model_layout.addWidget(self.select_model_btn)
        
        # Threshold setting
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.5)
        threshold_layout.addWidget(self.threshold_spin)
        model_layout.addLayout(threshold_layout)
        
        layout.addWidget(model_group)
        
        # Process button
        self.process_unet_btn = QPushButton("Run UNet Segmentation")
        self.process_unet_btn.clicked.connect(self.run_unet_processing)
        self.process_unet_btn.setEnabled(False)
        layout.addWidget(self.process_unet_btn)
        
        # Post-processing
        self.setup_postprocessing_controls(layout)
        
        layout.addStretch()
        self.tab_widget.addTab(unet_widget, "UNet")
        
    def setup_ilastik_tab(self):
        """Setup Ilastik processing tab"""
        ilastik_widget = QWidget()
        layout = QVBoxLayout(ilastik_widget)
        
        # Segmentation file selection
        seg_group = QGroupBox("Ilastik Segmentation")
        seg_layout = QVBoxLayout(seg_group)
        
        self.seg_path_label = QLabel("No segmentation selected")
        self.seg_path_label.setWordWrap(True)
        seg_layout.addWidget(self.seg_path_label)
        
        self.select_seg_btn = QPushButton("Select Segmentation")
        self.select_seg_btn.clicked.connect(self.select_segmentation)
        seg_layout.addWidget(self.select_seg_btn)
        
        layout.addWidget(seg_group)
        
        # Refinement settings
        refine_group = QGroupBox("Refinement Settings")
        refine_layout = QVBoxLayout(refine_group)
        
        # Dilation range
        dilation_layout = QHBoxLayout()
        dilation_layout.addWidget(QLabel("Dilation Range:"))
        self.dilation_min_spin = QSpinBox()
        self.dilation_min_spin.setRange(1, 10)
        self.dilation_min_spin.setValue(2)
        dilation_layout.addWidget(self.dilation_min_spin)
        dilation_layout.addWidget(QLabel("to"))
        self.dilation_max_spin = QSpinBox()
        self.dilation_max_spin.setRange(1, 10)
        self.dilation_max_spin.setValue(5)
        dilation_layout.addWidget(self.dilation_max_spin)
        refine_layout.addLayout(dilation_layout)
        
        layout.addWidget(refine_group)
        
        # Process button
        self.process_ilastik_btn = QPushButton("Refine Ilastik Output")
        self.process_ilastik_btn.clicked.connect(self.run_ilastik_processing)
        self.process_ilastik_btn.setEnabled(False)
        layout.addWidget(self.process_ilastik_btn)
        
        # Post-processing
        self.setup_postprocessing_controls(layout)
        
        layout.addStretch()
        self.tab_widget.addTab(ilastik_widget, "Ilastik")
        
    def setup_postprocessing_controls(self, parent_layout):
        """Setup common post-processing controls"""
        postprocess_group = QGroupBox("Post-Processing")
        layout = QVBoxLayout(postprocess_group)
        
        # Skeletonization
        skel_layout = QHBoxLayout()
        self.skeletonize_btn = QPushButton("Skeletonize")
        self.skeletonize_btn.clicked.connect(self.run_skeletonization)
        self.skeletonize_btn.setEnabled(False)
        skel_layout.addWidget(self.skeletonize_btn)
        
        self.skel_method_combo = QComboBox()
        self.skel_method_combo.addItems(["zhang", "lee"])
        skel_layout.addWidget(self.skel_method_combo)
        layout.addLayout(skel_layout)
        
        # Endpoint connection
        connect_layout = QHBoxLayout()
        self.connect_endpoints_btn = QPushButton("Connect Endpoints")
        self.connect_endpoints_btn.clicked.connect(self.run_endpoint_connection)
        self.connect_endpoints_btn.setEnabled(False)
        connect_layout.addWidget(self.connect_endpoints_btn)
        layout.addLayout(connect_layout)
        
        # Connection parameters
        params_layout = QVBoxLayout()
        
        max_dist_layout = QHBoxLayout()
        max_dist_layout.addWidget(QLabel("Max Distance:"))
        self.max_distance_spin = QSpinBox()
        self.max_distance_spin.setRange(5, 100)
        self.max_distance_spin.setValue(25)
        max_dist_layout.addWidget(self.max_distance_spin)
        params_layout.addLayout(max_dist_layout)
        
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Iterations:"))
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 10)
        self.iterations_spin.setValue(3)
        iter_layout.addWidget(self.iterations_spin)
        params_layout.addLayout(iter_layout)
        
        layout.addLayout(params_layout)
        parent_layout.addWidget(postprocess_group)
        
    def setup_editing_tab(self):
        """Setup manual editing tab"""
        edit_widget = QWidget()
        layout = QVBoxLayout(edit_widget)
        
        # Brush tools
        brush_group = QGroupBox("Brush Tools")
        brush_layout = QVBoxLayout(brush_group)
        
        # Tool selection
        tool_layout = QHBoxLayout()
        self.tool_group = QButtonGroup()
        
        self.draw_tool_btn = QToolButton()
        self.draw_tool_btn.setText("Draw")
        self.draw_tool_btn.setCheckable(True)
        self.draw_tool_btn.setChecked(True)
        self.tool_group.addButton(self.draw_tool_btn, 0)
        tool_layout.addWidget(self.draw_tool_btn)
        
        self.erase_tool_btn = QToolButton()
        self.erase_tool_btn.setText("Erase")
        self.erase_tool_btn.setCheckable(True)
        self.tool_group.addButton(self.erase_tool_btn, 1)
        tool_layout.addWidget(self.erase_tool_btn)
        
        brush_layout.addLayout(tool_layout)
        
        # Brush size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Brush Size:"))
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setRange(1, 50)
        self.brush_size_slider.setValue(5)
        size_layout.addWidget(self.brush_size_slider)
        
        self.brush_size_label = QLabel("5")
        size_layout.addWidget(self.brush_size_label)
        brush_layout.addLayout(size_layout)
        
        layout.addWidget(brush_group)
        
        # Editing controls
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo_edit)
        controls_layout.addWidget(self.undo_btn)
        
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.clicked.connect(self.redo_edit)
        controls_layout.addWidget(self.redo_btn)
        
        self.clear_edits_btn = QPushButton("Clear All Edits")
        self.clear_edits_btn.clicked.connect(self.clear_edits)
        controls_layout.addWidget(self.clear_edits_btn)
        
        layout.addWidget(controls_group)
        
        layout.addStretch()
        self.tab_widget.addTab(edit_widget, "Manual Edit")
        
    def setup_export_tab(self):
        """Setup export tab"""
        export_widget = QWidget()
        layout = QVBoxLayout(export_widget)
        
        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout(export_group)
        
        self.export_svg_btn = QPushButton("Export as SVG")
        self.export_svg_btn.clicked.connect(self.export_svg)
        self.export_svg_btn.setEnabled(False)
        export_layout.addWidget(self.export_svg_btn)
        
        self.export_mask_btn = QPushButton("Export Mask")
        self.export_mask_btn.clicked.connect(self.export_mask)
        self.export_mask_btn.setEnabled(False)
        export_layout.addWidget(self.export_mask_btn)
        
        layout.addWidget(export_group)
        
        layout.addStretch()
        self.tab_widget.addTab(export_widget, "Export")
        
    def setup_image_panel(self, parent):
        """Setup the right image panel"""
        image_frame = QFrame()
        image_frame.setFrameStyle(QFrame.StyledPanel)
        parent.addWidget(image_frame)
        
        layout = QVBoxLayout(image_frame)
        
        # Image viewer with brush editor
        self.image_viewer = ImageViewer()
        self.brush_editor = BrushEditor(self.image_viewer)
        layout.addWidget(self.image_viewer)
        
        # Display controls
        controls_layout = QHBoxLayout()
        
        self.show_original_cb = QCheckBox("Show Original")
        self.show_original_cb.setChecked(True)
        self.show_original_cb.toggled.connect(self.update_display)
        controls_layout.addWidget(self.show_original_cb)
        
        self.show_mask_cb = QCheckBox("Show Mask")
        self.show_mask_cb.setChecked(True)
        self.show_mask_cb.toggled.connect(self.update_display)
        controls_layout.addWidget(self.show_mask_cb)
        
        self.show_skeleton_cb = QCheckBox("Show Skeleton")
        self.show_skeleton_cb.setChecked(True)
        self.show_skeleton_cb.toggled.connect(self.update_display)
        controls_layout.addWidget(self.show_skeleton_cb)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
    def setup_log_window(self, parent_layout):
        """Setup log window"""
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        parent_layout.addWidget(log_group)
        
    def setup_menus(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_action = QAction('Open Image', self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.select_image)
        file_menu.addAction(open_action)
        
        save_action = QAction('Save SVG', self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.export_svg)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
    def setup_connections(self):
        """Setup signal connections"""
        # Brush editor connections
        self.brush_size_slider.valueChanged.connect(self.update_brush_size)
        self.tool_group.buttonClicked.connect(self.change_tool)
        
        # Brush editor signals
        self.brush_editor.mask_changed.connect(self.on_mask_edited)
        self.brush_editor.skeleton_changed.connect(self.on_skeleton_edited)
        
    def clear_all_data(self):
        """Clear all current data"""
        self.current_mask = None
        self.current_skeleton = None
        self.probability_map = None
        
        # Clear viewer
        self.image_viewer.set_mask(None)
        self.image_viewer.set_skeleton(None)
        
        # Clear brush editor
        self.brush_editor.set_mask(None)
        self.brush_editor.set_skeleton(None)
        
        # Disable buttons
        self.skeletonize_btn.setEnabled(False)
        self.connect_endpoints_btn.setEnabled(False)
        self.export_svg_btn.setEnabled(False)
        self.export_mask_btn.setEnabled(False)
        
        self.log("Cleared all existing data")
        
    # File operations
    @Slot()
    def select_image(self):
        """Select input image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.tiff *.tif *.bmp)"
        )
        
        if file_path:
            try:
                # Clear existing data when loading new image
                self.clear_all_data()
                
                from PIL import Image
                image = Image.open(file_path)
                self.original_image = np.array(image)
                
                # Crop to square for display (same as neural network preprocessing)
                self.display_image = crop_image_to_square(self.original_image)
                
                print(f"MainWindow: Original image shape: {self.original_image.shape}")
                print(f"MainWindow: Display image shape: {self.display_image.shape}")
                
                self.image_path_label.setText(f"Selected: {Path(file_path).name}")
                self.image_viewer.set_image(self.display_image)  # Use cropped image
                self.brush_editor.set_image(self.display_image)
                
                # Enable processing
                self.process_unet_btn.setEnabled(True)
                
                self.log(f"Loaded image: {file_path}")
                self.log(f"Original size: {self.original_image.shape}, Display size: {self.display_image.shape}")
                
            except Exception as e:
                self.log(f"Error loading image: {str(e)}")
                
    @Slot()
    def select_model(self):
        """Select UNet model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", "", "PyTorch Models (*.pth *.pt)"
        )
        
        if file_path:
            self.model_path_label.setText(f"Selected: {Path(file_path).name}")
            self.model_path = file_path
            self.log(f"Selected model: {file_path}")
            
    @Slot()
    def select_segmentation(self):
        """Select Ilastik segmentation"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Segmentation", "", 
            "Image Files (*.png *.jpg *.jpeg *.tiff *.tif)"
        )
        
        if file_path:
            try:
                # Clear existing data when loading segmentation
                self.clear_all_data()
                
                # Load the segmentation as the image
                from PIL import Image
                seg_image = Image.open(file_path)
                seg_array = np.array(seg_image)
                
                # If it's a color image, convert to grayscale
                if len(seg_array.shape) == 3:
                    seg_array = np.mean(seg_array, axis=2)
                
                # Crop to square
                self.display_image = crop_image_to_square(seg_array)
                
                self.seg_path_label.setText(f"Selected: {Path(file_path).name}")
                self.segmentation_path = file_path
                
                # Set the segmentation as the working image
                self.image_viewer.set_image(self.display_image)
                self.brush_editor.set_image(self.display_image)
                
                self.process_ilastik_btn.setEnabled(True)
                self.log(f"Selected segmentation: {file_path}")
                
            except Exception as e:
                self.log(f"Error loading segmentation: {str(e)}")
                
    # Processing operations
    @Slot()
    def run_unet_processing(self):
        """Run UNet processing"""
        if not hasattr(self, 'model_path') or self.display_image is None:
            self.log("Please select both image and model first")
            return
            
        self.progress_bar.setVisible(True)
        self.process_unet_btn.setEnabled(False)
        
        # Pass the display image (square cropped) to the neural network
        self.nn_processor = NeuralNetworkProcessor(
            self.display_image,  # Use cropped image
            self.model_path, 
            self.threshold_spin.value()
        )
        self.nn_processor.finished.connect(self.on_unet_finished)
        self.nn_processor.progress.connect(self.log)
        self.nn_processor.start()
        
    @Slot()
    def run_ilastik_processing(self):
        """Run Ilastik processing"""
        if not hasattr(self, 'segmentation_path'):
            self.log("Please select segmentation first")
            return
            
        self.progress_bar.setVisible(True)
        self.process_ilastik_btn.setEnabled(False)
        
        dilation_range = (self.dilation_min_spin.value(), self.dilation_max_spin.value())
        
        self.ilastik_refiner = IlastikRefiner(self.segmentation_path, dilation_range)
        self.ilastik_refiner.finished.connect(self.on_ilastik_finished)
        self.ilastik_refiner.progress.connect(self.log)
        self.ilastik_refiner.start()
        
    @Slot()
    def run_skeletonization(self):
        """Run skeletonization"""
        if self.current_mask is None:
            self.log("No mask available for skeletonization")
            return
            
        self.progress_bar.setVisible(True)
        self.skeletonize_btn.setEnabled(False)
        
        self.skeletonizer = Skeletonizer(
            self.current_mask, 
            self.skel_method_combo.currentText()
        )
        self.skeletonizer.finished.connect(self.on_skeletonization_finished)
        self.skeletonizer.progress.connect(self.log)
        self.skeletonizer.start()
        
    @Slot()
    def run_endpoint_connection(self):
        """Run endpoint connection"""
        if self.current_skeleton is None:
            self.log("No skeleton available for endpoint connection")
            return
            
        self.progress_bar.setVisible(True)
        self.connect_endpoints_btn.setEnabled(False)
        
        self.endpoint_connector = EndpointConnector(
            self.current_skeleton,
            self.max_distance_spin.value(),
            self.iterations_spin.value()
        )
        self.endpoint_connector.finished.connect(self.on_endpoint_connection_finished)
        self.endpoint_connector.progress.connect(self.log)
        self.endpoint_connector.start()
        
    # Processing callbacks
    @Slot()
    def on_mask_edited(self):
        """Handle mask being edited"""
        self.current_mask = self.brush_editor.mask
        self.update_export_buttons()
        
    @Slot()
    def on_skeleton_edited(self):
        """Handle skeleton being edited"""
        self.current_skeleton = self.brush_editor.skeleton
        self.update_export_buttons()
        
    @Slot(np.ndarray, np.ndarray)
    def on_unet_finished(self, mask, probability_map):
        """Handle UNet processing completion"""
        print(f"MainWindow: UNet finished - mask shape: {mask.shape}, prob shape: {probability_map.shape}")
        
        self.current_mask = mask
        self.probability_map = probability_map
        
        print(f"MainWindow: Setting mask in image viewer...")
        self.image_viewer.set_mask(mask)
        self.brush_editor.set_mask(mask)
        
        self.progress_bar.setVisible(False)
        self.process_unet_btn.setEnabled(True)
        self.skeletonize_btn.setEnabled(True)
        
        self.update_display()
        self.update_export_buttons()
        self.log("UNet processing completed")
        
    @Slot(np.ndarray, dict)
    def on_ilastik_finished(self, refined_mask, analysis_results):
        """Handle Ilastik processing completion"""
        self.current_mask = refined_mask
        self.image_viewer.set_mask(refined_mask)
        self.brush_editor.set_mask(refined_mask)
        
        # If skeleton was found, use it
        if 'best_result' in analysis_results:
            self.current_skeleton = analysis_results['best_result']['skeleton']
            self.image_viewer.set_skeleton(self.current_skeleton)
            self.brush_editor.set_skeleton(self.current_skeleton)
            self.connect_endpoints_btn.setEnabled(True)
            
        self.progress_bar.setVisible(False)
        self.process_ilastik_btn.setEnabled(True)
        self.skeletonize_btn.setEnabled(True)
        
        self.update_display()
        self.update_export_buttons()
        self.log("Ilastik refinement completed")
        
    @Slot(np.ndarray)
    def on_skeletonization_finished(self, skeleton):
        """Handle skeletonization completion"""
        print(f"MainWindow: Skeletonization finished - skeleton shape: {skeleton.shape}")
        
        self.current_skeleton = skeleton
        self.image_viewer.set_skeleton(skeleton)
        self.brush_editor.set_skeleton(skeleton)
        
        self.progress_bar.setVisible(False)
        self.skeletonize_btn.setEnabled(True)
        self.connect_endpoints_btn.setEnabled(True)
        
        self.update_display()
        self.update_export_buttons()
        self.log("Skeletonization completed")
        
    @Slot(np.ndarray, list)
    def on_endpoint_connection_finished(self, connected_skeleton, connections):
        """Handle endpoint connection completion"""
        self.current_skeleton = connected_skeleton
        self.image_viewer.set_skeleton(connected_skeleton)
        self.brush_editor.set_skeleton(connected_skeleton)
        
        self.progress_bar.setVisible(False)
        self.connect_endpoints_btn.setEnabled(True)
        
        self.update_display()
        self.update_export_buttons()
        self.log(f"Endpoint connection completed. Made {len(connections)} connections")
        
    # Manual editing
    @Slot(int)
    def update_brush_size(self, size):
        """Update brush size"""
        self.brush_size_label.setText(str(size))
        self.brush_editor.set_brush_size(size)
        
    @Slot()
    def change_tool(self, button):
        """Change editing tool"""
        tool_id = self.tool_group.id(button)
        if tool_id == 0:  # Draw
            self.brush_editor.set_tool('draw')
        elif tool_id == 1:  # Erase
            self.brush_editor.set_tool('erase')
            
    @Slot()
    def undo_edit(self):
        """Undo last edit"""
        self.brush_editor.undo()
        
    @Slot()
    def redo_edit(self):
        """Redo last edit"""
        self.brush_editor.redo()
        
    @Slot()
    def clear_edits(self):
        """Clear all edits"""
        self.brush_editor.clear_edits()
        
    # Display
    @Slot()
    def update_display(self):
        """Update image display"""
        print(f"MainWindow: Updating display - show_mask: {self.show_mask_cb.isChecked()}")
        self.image_viewer.set_display_options(
            show_original=self.show_original_cb.isChecked(),
            show_mask=self.show_mask_cb.isChecked(),
            show_skeleton=self.show_skeleton_cb.isChecked()
        )
        
    @Slot()
    def update_export_buttons(self):
        """Update export button states"""
        has_skeleton = self.current_skeleton is not None and bool(np.any(self.current_skeleton > 0))
        has_mask = self.current_mask is not None and bool(np.any(self.current_mask > 0))
        
        self.export_svg_btn.setEnabled(has_skeleton)
        self.export_mask_btn.setEnabled(has_mask)
        
    # Export
    @Slot()
    def export_svg(self):
        """Export skeleton as SVG"""
        if self.current_skeleton is None or not bool(np.any(self.current_skeleton > 0)):
            self.log("No skeleton to export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export SVG", "skeleton.svg", "SVG Files (*.svg)"
        )
        
        if file_path:
            try:
                # Ensure .svg extension
                if not file_path.lower().endswith('.svg'):
                    file_path += '.svg'
                    
                exporter = SVGExporter()
                image_shape = self.display_image.shape if self.display_image is not None else None
                exporter.export_skeleton_to_svg(
                    self.current_skeleton, 
                    file_path, 
                    image_shape
                )
                self.log(f"SVG exported to: {file_path}")
            except Exception as e:
                self.log(f"Error exporting SVG: {str(e)}")
                import traceback
                traceback.print_exc()
                
    @Slot()
    def export_mask(self):
        """Export current mask"""
        if self.current_mask is None or not bool(np.any(self.current_mask > 0)):
            self.log("No mask to export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Mask", "mask.png", "PNG Files (*.png);;TIFF Files (*.tiff);;All Files (*)"
        )
        
        if file_path:
            try:
                # Ensure proper file extension
                if not any(file_path.lower().endswith(ext) for ext in ['.png', '.tiff', '.tif', '.jpg', '.jpeg']):
                    file_path += '.png'
                    
                from PIL import Image
                mask_image = Image.fromarray((self.current_mask * 255).astype(np.uint8))
                mask_image.save(file_path)
                self.log(f"Mask exported to: {file_path}")
            except Exception as e:
                self.log(f"Error exporting mask: {str(e)}")
                import traceback
                traceback.print_exc()
                
    def log(self, message):
        """Add message to log"""
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()

def main():
    app = QApplication(sys.argv)
    window = CellTracingMainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()