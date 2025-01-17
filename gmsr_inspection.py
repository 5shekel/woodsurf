import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io, color, img_as_float, measure
from skimage.morphology import binary_dilation
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
try:
    import cupy as cp
    HAS_GPU = True
    print("GPU acceleration enabled")
except ImportError:
    HAS_GPU = False
    print("GPU acceleration not available. Install CuPy for GPU support.")

def compute_gaussian_derivative_filters(tau_g=2.0):
    """
    Compute Gaussian derivative filters as per equations (1) and (2) in the paper.
    """
    x = np.arange(-3*tau_g, 3*tau_g + 1)
    y = np.arange(-3*tau_g, 3*tau_g + 1)
    X, Y = np.meshgrid(x, y)
    
    gx = -X / (2 * np.pi * tau_g**4) * np.exp(-(X**2 + Y**2)/(2 * tau_g**2))
    gy = -Y / (2 * np.pi * tau_g**4) * np.exp(-(X**2 + Y**2)/(2 * tau_g**2))
    
    if HAS_GPU:
        return cp.asarray(gx), cp.asarray(gy)
    return gx, gy

def process_region(args):
    """Process a single region for parallel execution"""
    region, low_mask = args
    grown = region.copy()
    iteration = 0
    while True:
        dilated = binary_dilation(grown)
        new_grown = dilated & low_mask
        if np.array_equal(new_grown, grown) or iteration > 100:
            break
        grown = new_grown
        iteration += 1
    return grown

def extract_gmsr(image, tau_g=2.0, high_threshold=0.2, low_threshold=0.15, num_cores=8):
    """
    Extract Gradient Magnitude based Support Regions with GPU acceleration
    Args:
        image: Input image
        tau_g: Scale parameter for Gaussian derivatives
        high_threshold: Higher threshold for hysteresis
        low_threshold: Lower threshold for hysteresis
        num_cores: Number of CPU cores to use for parallel processing (default: 8)
    """
    if HAS_GPU:
        # Move image to GPU
        image_gpu = cp.asarray(image)
        
        # Get Gaussian derivative filters (already on GPU)
        gx, gy = compute_gaussian_derivative_filters(tau_g)
        
        # Compute gradients on GPU
        Gx = cp.array(ndimage.convolve(cp.asnumpy(image_gpu), cp.asnumpy(gx), mode='reflect'))
        Gy = cp.array(ndimage.convolve(cp.asnumpy(image_gpu), cp.asnumpy(gy), mode='reflect'))
        
        # Compute magnitude on GPU
        M = cp.sqrt(Gx**2 + Gy**2)
        M = M / cp.max(M)
        
        # Move back to CPU for thresholding
        M = cp.asnumpy(M)
    else:
        # CPU-only computation
        gx, gy = compute_gaussian_derivative_filters(tau_g)
        Gx = ndimage.convolve(image, gx, mode='reflect')
        Gy = ndimage.convolve(image, gy, mode='reflect')
        M = np.sqrt(Gx**2 + Gy**2)
        M = M / np.max(M)

    # Thresholding
    high_mask = M > high_threshold
    low_mask = M > low_threshold
    
    # Label regions
    labels, num_labels = ndimage.label(high_mask)
    print(f"Found {num_labels} initial regions")
    
    # Prepare regions for parallel processing
    regions = [(labels == label, low_mask) for label in range(1, num_labels + 1)]
    
    # Use specified number of cores
    num_cores = min(num_cores, multiprocessing.cpu_count())
    print(f"Processing regions using {num_cores} CPU cores...")
    
    SR = np.zeros_like(M, dtype=bool)
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        chunk_size = max(1, len(regions) // (num_cores * 4))  # Process regions in chunks
        for i, grown in enumerate(executor.map(process_region, regions, chunksize=chunk_size)):
            if i % 10 == 0:
                print(f"Processing region {i+1}/{num_labels}")
            SR |= grown
    
    return SR

def save_as_svg(binary_mask, output_path, scale=1.0):
    """
    Convert binary mask to SVG paths and save to file
    Args:
        binary_mask: Binary mask from GMSR
        output_path: Path to save SVG file
        scale: Scale factor for SVG coordinates
    """
    # Find contours in the binary mask
    contours = measure.find_contours(binary_mask, 0.5)
    
    # SVG header with viewBox matching the image dimensions
    height, width = binary_mask.shape
    svg_content = f'<?xml version="1.0" encoding="UTF-8"?>\n'
    svg_content += f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">\n'
    
    # Convert each contour to SVG path
    for contour in contours:
        # Scale coordinates
        contour = contour * scale
        # Start path
        path = "M"
        # Add each point
        for x, y in contour:
            path += f" {y:.1f},{x:.1f}"
        path += " Z"  # Close path
        
        # Add path to SVG with styling
        svg_content += f'  <path d="{path}" fill="none" stroke="black" stroke-width="1"/>\n'
    
    svg_content += '</svg>'
    
    # Save SVG file
    with open(output_path, 'w') as f:
        f.write(svg_content)
    print(f"Saved SVG contours to {output_path}")

def process_image(image_path, crop_x=200, crop_y=400, crop_width=900, crop_height=900, output_svg=None, num_cores=8):
    """
    Process image and extract GMSR features with GPU acceleration
    Args:
        image_path: Path to the image
        crop_x: X coordinate of crop start position
        crop_y: Y coordinate of crop start position
        crop_width: Width of crop region
        crop_height: Height of crop region
        output_svg: Path to save SVG output (optional)
        num_cores: Number of CPU cores to use (default: 8)
    """
    print("Loading image...")
    image = img_as_float(io.imread(image_path))
    if image.ndim == 3:
        image = color.rgb2gray(image)
    
    print(f"Cropping image to region: x={crop_x}, y={crop_y}, width={crop_width}, height={crop_height}")
    end_y = min(crop_y + crop_height, image.shape[0])
    end_x = min(crop_x + crop_width, image.shape[1])
    image = image[crop_y:end_y, crop_x:end_x]
    print(f"Cropped image shape: {image.shape}")
    
    print("Extracting GMSR...")
    gmsr = extract_gmsr(image, num_cores=num_cores)
    print("GMSR extraction complete")
    
    # Save GMSR features as SVG if requested
    if output_svg:
        print(f"Converting GMSR features to SVG...")
        save_as_svg(gmsr, output_svg)
    
    # Display visualization
    print("Visualizing results...")
    fig = plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.title('Original Image (Cropped)')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(122)
    plt.title('GMSR Features')
    plt.imshow(gmsr, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Visualization complete")
    
    return gmsr

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python gmsr_inspection.py <image_path> [output_svg] [num_cores]")
        sys.exit(1)
    
    multiprocessing.freeze_support()  # Required for Windows
    image_path = sys.argv[1]
    output_svg = sys.argv[2] if len(sys.argv) > 2 else None
    num_cores = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    
    process_image(image_path, 
                 crop_x=200, 
                 crop_y=400, 
                 crop_width=900, 
                 crop_height=900,
                 output_svg=output_svg,
                 num_cores=num_cores)
