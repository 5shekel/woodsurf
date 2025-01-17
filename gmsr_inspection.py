import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io, color, img_as_float
from skimage.morphology import binary_dilation
import cmath

def compute_gaussian_derivative_filters(tau_g=2.0):
    """
    Compute Gaussian derivative filters as per equations (1) and (2) in the paper.
    tau_g is the scale parameter, set to 2.0 as per paper's appendix.
    """
    # Create coordinate grid
    x = np.arange(-3*tau_g, 3*tau_g + 1)
    y = np.arange(-3*tau_g, 3*tau_g + 1)
    X, Y = np.meshgrid(x, y)
    
    # Equation (1): gx filter
    gx = -X / (2 * np.pi * tau_g**4) * np.exp(-(X**2 + Y**2)/(2 * tau_g**2))
    
    # Equation (2): gy filter
    gy = -Y / (2 * np.pi * tau_g**4) * np.exp(-(X**2 + Y**2)/(2 * tau_g**2))
    
    return gx, gy

def extract_gmsr(image, tau_g=2.0, high_threshold=0.2, low_threshold=0.15):
    """
    Extract Gradient Magnitude based Support Regions as described in Section 2.1
    """
    print("Computing Gaussian derivative filters...")
    # Get Gaussian derivative filters
    gx, gy = compute_gaussian_derivative_filters(tau_g)
    
    print("Computing gradient magnitudes...")
    # Compute smoothed gradients using convolution
    Gx = ndimage.convolve(image, gx, mode='reflect')
    Gy = ndimage.convolve(image, gy, mode='reflect')
    
    # Compute gradient magnitude M(x,y)
    M = np.sqrt(Gx**2 + Gy**2)
    
    # Normalize gradient magnitude
    M = M / np.max(M)
    
    print("Applying hysteresis thresholding...")
    # Apply hysteresis thresholding as described in the paper
    high_mask = M > high_threshold
    low_mask = M > low_threshold
    
    # Label high threshold regions
    labels, num_labels = ndimage.label(high_mask)
    print(f"Found {num_labels} initial regions")
    
    # Grow regions using hysteresis
    SR = np.zeros_like(M, dtype=bool)
    for i, label in enumerate(range(1, num_labels + 1)):
        if i % 10 == 0:  # Print progress every 10 regions
            print(f"Processing region {i+1}/{num_labels}")
        region = labels == label
        # Grow region to neighboring pixels above low threshold
        grown = region.copy()
        iteration = 0
        while True:
            dilated = binary_dilation(grown)
            new_grown = dilated & low_mask
            if np.array_equal(new_grown, grown) or iteration > 100:  # Add iteration limit
                break
            grown = new_grown
            iteration += 1
        SR |= grown
    
    return SR

def compute_log_filter(tau_l=1.0):
    """
    Compute Laplacian of Gaussian filter as per equation (3) in the paper.
    tau_l is the scale parameter, set to 1.0 as per paper.
    """
    # Create coordinate grid
    x = np.arange(-3*tau_l, 3*tau_l + 1)
    y = np.arange(-3*tau_l, 3*tau_l + 1)
    X, Y = np.meshgrid(x, y)
    R2 = X**2 + Y**2
    
    # Equation (3): LoG filter
    f = (1 / (np.pi * tau_l**4)) * (R2/(2*tau_l**2) - 1) * np.exp(-R2/(2*tau_l**2))
    
    return f

def extract_lgsr(image, tau_l=1.0, threshold=0.2):
    """
    Extract Laplacian of Gaussian based Support Regions as described in Section 2.2
    """
    print("Computing LoG filter...")
    # Get LoG filter
    f = compute_log_filter(tau_l)
    
    print("Applying LoG filter...")
    # Convolve image with LoG filter
    M = ndimage.convolve(image, f, mode='reflect')
    
    # Normalize response
    M = M / np.max(np.abs(M))
    
    print("Thresholding LoG response...")
    # Threshold for blob detection
    SR = np.abs(M) > threshold
    
    return SR

def extract_boundary_curve(support_region):
    """
    Extract boundary curve representation as described in Section 2.4
    Returns complex periodic function b(t) = bx(t) + jby(t)
    """
    # Get boundary points
    boundary = np.zeros_like(support_region, dtype=bool)
    boundary[1:-1, 1:-1] = support_region[1:-1, 1:-1] & ~(
        support_region[:-2, 1:-1] & 
        support_region[2:, 1:-1] & 
        support_region[1:-1, :-2] & 
        support_region[1:-1, 2:]
    )
    
    # Convert to complex periodic function
    y, x = np.nonzero(boundary)
    if len(x) == 0:
        return None
        
    # Create complex representation b(t)
    b = x + 1j*y
    
    # Sort points to form continuous boundary
    ordered = []
    current = b[0]
    used = set()
    
    while len(ordered) < len(b):
        ordered.append(current)
        used.add(current)
        
        # Find nearest unused point
        distances = np.abs(b - current)
        for next_point in b[np.argsort(distances)]:
            if next_point not in used:
                current = next_point
                break
    
    return np.array(ordered)

def compute_fourier_coefficients(b, order=4):
    """
    Compute Fourier series coefficients as per equations (4) and (5)
    """
    T = len(b)
    Bn = []
    
    for n in range(order + 1):
        coef = np.sum(b * np.exp(-2j * np.pi * n * np.arange(T) / T)) / T
        Bn.append(coef)
    
    return np.array(Bn)

def compute_curvature(b_hat):
    """
    Compute curvature as per equation (6)
    """
    # Compute derivatives
    t = np.arange(len(b_hat))
    db_dt = np.gradient(b_hat)
    d2b_dt2 = np.gradient(db_dt)
    
    # Extract real and imaginary parts
    dx_dt = db_dt.real
    dy_dt = db_dt.imag
    d2x_dt2 = d2b_dt2.real
    d2y_dt2 = d2b_dt2.imag
    
    # Compute curvature
    numerator = dx_dt * d2y_dt2 - dy_dt * d2x_dt2
    denominator = (dx_dt**2 + dy_dt**2)**(3/2)
    
    return numerator / (denominator + 1e-10)  # Add small constant to avoid division by zero

def process_image(image_path, crop_x=200, crop_y=400, crop_width=900, crop_height=900):
    """
    Process image following the paper's methodology
    Args:
        image_path: Path to the image
        crop_x: X coordinate of crop start position
        crop_y: Y coordinate of crop start position
        crop_width: Width of crop region
        crop_height: Height of crop region
    """
    print("Loading image...")
    # Load and convert image to grayscale
    image = img_as_float(io.imread(image_path))
    if image.ndim == 3:
        image = color.rgb2gray(image)
    
    # Crop the image to the specified region
    print(f"Cropping image to region: x={crop_x}, y={crop_y}, width={crop_width}, height={crop_height}")
    # Ensure crop coordinates are within image bounds
    end_y = min(crop_y + crop_height, image.shape[0])
    end_x = min(crop_x + crop_width, image.shape[1])
    image = image[crop_y:end_y, crop_x:end_x]
    print(f"Cropped image shape: {image.shape}")
    
    print("Extracting GMSR...")
    # Extract GMSR
    gmsr = extract_gmsr(image)
    print("GMSR extraction complete")
    
    print("Extracting LGSR...")
    # Extract LGSR
    lgsr = extract_lgsr(image)
    print("LGSR extraction complete")
    
    print("Visualizing results...")
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.title('Original Image (Cropped)')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(132)
    plt.title('GMSR')
    plt.imshow(gmsr, cmap='gray')
    plt.axis('off')
    
    plt.subplot(133)
    plt.title('LGSR')
    plt.imshow(lgsr, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Visualization complete")
    
    return gmsr, lgsr

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python gmsr_inspection.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    # Process with specified crop region
    process_image(image_path, crop_x=200, crop_y=400, crop_width=900, crop_height=900)
