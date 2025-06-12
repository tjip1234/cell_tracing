# Cell Tracing Application

A comprehensive tool for cell image analysis and tracing using neural networks, manual editing, and automated processing pipelines.

## Features

- **Neural Network Processing**: UNet-based cell segmentation
- **Ilastik Integration**: Advanced segmentation refinement
- **Manual Editing**: Interactive brush tools for mask and skeleton editing
- **Skeletonization**: Multiple algorithms including gentle, zhang, lee, and watershed methods
- **Zoom & Pan**: Smooth zooming with Ctrl+Mouse Wheel, fit-to-window functionality
- **SVG Export**: Export processed results as scalable vector graphics
- **Endpoint Connection**: Automatic connection of skeleton fragments

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Quick Setup

1. **Clone or download the project**:
   ```bash
   git clone https://github.com/tjip1234/cell_tracing
   cd cell_tracing
   ```
   
   Or download and extract the ZIP file to a folder called `cell_tracing`

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   
   **On Linux/Mac**:
   ```bash
   source .venv/bin/activate
   ```
   
   **On Windows**:
   ```bash
   .venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**:
   ```bash
   python run_app.py
   ```

### Manual Installation

If you prefer to install packages individually:

```bash
# GUI Framework
pip install PySide6

# Image Processing
pip install numpy opencv-python scikit-image scipy Pillow

# Machine Learning (for neural network features)
pip install torch torchvision
```

## Usage

### Basic Workflow

1. **Load an Image**: 
   - Click "Select Image" or use File → Open Image
   - Supports common formats (PNG, JPG, TIFF)

2. **Process with Neural Network**:
   - Go to "Neural Network" tab
   - Set threshold (0.5 recommended)
   - Click "Process with UNet"

3. **Manual Editing** (kinda sucks currently):
   - Switch to "Manual Edit" tab
   - Use brush tools to refine the mask
   - Adjust brush size with slider
   - Switch between Draw/Erase tools

4. **Skeletonization**:
   - Choose method: "gentle", "zhang"(recommended), "lee", or "watershed"
   - Set smoothing level: "light", "medium", "heavy"(recommended), or "none"
   - Click "Skeletonize Current Mask"
5. **Connect Endpoints**:
   - Only useful for zhang and lee methods
   - Default settings are usually fine
6. **Export Results**:
   - Go to "Export" tab
   - Click "Export SVG" to save results and refine manually later

### Controls

- **Zoom**: Ctrl + Mouse Wheel, or use zoom buttons
- **Pan**: Click and drag when zoomed in
- **Brush Size**: Adjust with slider in Manual Edit tab
- **Display Toggle**: Use checkboxes to show/hide original, mask, skeleton

### Keyboard Shortcuts

- `Ctrl + Plus`: Zoom In
- `Ctrl + Minus`: Zoom Out
- `Ctrl + 0`: Fit to Window
- `Ctrl + 1`: Reset Zoom (100%)
- `Ctrl + O`: Open Image
- `Ctrl + S`: Save SVG

## Troubleshooting

### Common Issues

1. **Application won't start**:
   - Ensure Python 3.8+ is installed
   - Check that virtual environment is activated
   - Verify all dependencies are installed: `pip list`

2. **Neural network features not working**:
   - Ensure PyTorch is properly installed
   - For GPU support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

3. **Image loading issues**:
   - Check image format is supported
   - Ensure image file is not corrupted
   - Try converting to PNG format

4. **Memory issues with large images**:
   - Consider resizing images before processing
   - Close other applications to free memory

### Performance Tips

- **For better skeletonization**: Use "zhang" with "heavy" smoothing and then use the connect endpoints option

## File Structure

```
cell_tracing/
├── main_window.py          # Main application window
├── run_app.py             # Application launcher
├── requirements.txt       # Package dependencies
├── widgets/
│   ├── image_viewer.py    # Image display with zoom/pan
│   └── brush_editor.py    # Manual editing tools
└── processing/
    ├── neural_network.py     # UNet processing
    ├── ilastik_refiner.py   # Ilastik integration
    ├── skeletonizer.py      # Skeletonization algorithms
    ├── endpoint_connector.py # Skeleton fragment connection
    └── svg_exporter.py      # SVG export functionality
```

## Requirements

See `requirements.txt` for complete list. Key dependencies:

- **PySide6**: GUI framework
- **NumPy**: Array operations
- **OpenCV**: Image processing
- **scikit-image**: Advanced image analysis
- **SciPy**: Scientific computing
- **Pillow**: Image I/O
- **PyTorch**: Neural network inference

## License

[Add your license information here]

## Development

To contribute or modify:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test it (if you want to lol)
5. Submit a pull request

