#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

from colour import read_image, matrix_colour_correction
from colour_checker_detection import detect_colour_checkers_segmentation

# For robust regression
from sklearn.linear_model import RANSACRegressor, LinearRegression

###############################
# Utility Functions
###############################

def get_image_files(folder):
    """
    Return a sorted list of image file paths matching both *.tif and *.TIF.
    """
    files_lower = glob.glob(os.path.join(folder, "*.tif"))
    files_upper = glob.glob(os.path.join(folder, "*.TIF"))
    return sorted(files_lower + files_upper)

def reshape_patches(patch_values):
    """
    Reshape patch values into a 2D grid.
    For a standard 24-patch ColorChecker, reshape to 4 rows x 6 columns.
    Otherwise, display as a vertical strip.
    """
    if patch_values.shape[0] == 24:
        return patch_values.reshape(4, 6, 3)
    else:
        return patch_values.reshape(-1, 1, 3)

def visualize_checker_detection_both(nikon_image, nikon_patches, nikon_filename,
                                     hasselblad_image, hasselblad_patches, hasselblad_filename):
    """
    Display a 2x2 figure showing both Nikon and Hasselblad detections.
    
    Top row: Original images.
    Bottom row: Detected patch grids.
    """
    nikon_patch_grid = reshape_patches(nikon_patches)
    hasselblad_patch_grid = reshape_patches(hasselblad_patches)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].imshow(nikon_image)
    axes[0, 0].set_title(f"Nikon: {os.path.basename(nikon_filename)}")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(hasselblad_image)
    axes[0, 1].set_title(f"Hasselblad: {os.path.basename(hasselblad_filename)}")
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(nikon_patch_grid)
    axes[1, 0].set_title("Nikon Detected Patches")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(hasselblad_patch_grid)
    axes[1, 1].set_title("Hasselblad Detected Patches (Exposure Calibrated)")
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    plt.show()

def calibrate_exposure_using_bottom_row(nikon_patches, hasselblad_patches):
    """
    For a standard 24-patch ColorChecker (4 rows x 6 columns), assume the bottom row
    (patches 19 to 24, indices 18:24) is neutral.
    Compute the mean intensity for that row for both Nikon and Hasselblad and
    return the exposure factor (nikon_mean / hasselblad_mean).
    
    If the patch count is not 24, fall back to using the patch with the lowest std.
    """
    if nikon_patches.shape[0] == 24 and hasselblad_patches.shape[0] == 24:
        nikon_grid = nikon_patches.reshape(4, 6, 3)
        hasselblad_grid = hasselblad_patches.reshape(4, 6, 3)
        nikon_bottom = nikon_grid[3, :, :]
        hasselblad_bottom = hasselblad_grid[3, :, :]
        nikon_mean = np.mean(nikon_bottom)
        hasselblad_mean = np.mean(hasselblad_bottom)
    else:
        nikon_idx = int(np.argmin(np.std(nikon_patches, axis=1)))
        hasselblad_idx = int(np.argmin(np.std(hasselblad_patches, axis=1)))
        nikon_mean = np.mean(nikon_patches[nikon_idx])
        hasselblad_mean = np.mean(hasselblad_patches[hasselblad_idx])
    if hasselblad_mean == 0:
        return 1.0
    return nikon_mean / hasselblad_mean

def gather_patch_data(nikon_files, hasselblad_files, visualize_detection=False):
    """
    For each Nikon/Hasselblad image pair, detect the ColorChecker and collect patch values.
    For each pair, adjust the Hasselblad patches for exposure using the bottom row average.
    
    Returns:
      X : ndarray of shape (total_patches, 3) for Nikon (source)
      Y : ndarray of shape (total_patches, 3) for Hasselblad (target, exposure corrected)
    """
    source_list = []
    target_list = []
    for nikon_file, hasselblad_file in zip(nikon_files, hasselblad_files):
        nikon_image = read_image(nikon_file)
        hasselblad_image = read_image(hasselblad_file)
        nikon_checkers = detect_colour_checkers_segmentation(nikon_image)
        hasselblad_checkers = detect_colour_checkers_segmentation(hasselblad_image)
        if len(nikon_checkers) != 1 or len(hasselblad_checkers) != 1:
            print(f"Skipping {nikon_file} / {hasselblad_file} due to detection issues.")
            continue
        nikon_patches = np.array(nikon_checkers[0])
        hasselblad_patches = np.array(hasselblad_checkers[0])
        if nikon_patches.shape != hasselblad_patches.shape:
            print(f"Skipping {nikon_file} / {hasselblad_file} due to mismatched patch counts.")
            continue
        exposure_factor = calibrate_exposure_using_bottom_row(nikon_patches, hasselblad_patches)
        hasselblad_patches = np.clip(hasselblad_patches * exposure_factor, 0.0, 1.0)
        if visualize_detection:
            visualize_checker_detection_both(nikon_image, nikon_patches, nikon_file,
                                             hasselblad_image, hasselblad_patches, hasselblad_file)
        source_list.append(nikon_patches)
        target_list.append(hasselblad_patches)
    if not source_list:
        raise Exception("No valid calibration pairs found!")
    X = np.vstack(source_list)
    Y = np.vstack(target_list)
    return X, Y

def compute_automatic_exposure_gain(X, Y, transform, robust=False):
    """
    Compute an automatic exposure gain factor based on the checker data.
    For standard method, predicted Y is computed as:
         Y_pred = X @ transform.T
    For robust method, X is first augmented with ones:
         Y_pred = [X, ones] @ transform.T
    The gain is calculated as:
         gain = mean(Y) / mean(Y_pred)
    """
    if robust:
        n = X.shape[0]
        X_aug = np.hstack([X, np.ones((n, 1))])
        Y_pred = np.dot(X_aug, transform.T)
    else:
        Y_pred = np.dot(X, transform.T)
    mean_target = np.mean(Y)
    mean_pred = np.mean(Y_pred)
    if mean_pred == 0:
        return 1.0
    return mean_target / mean_pred

###############################
# Standard Transformation
###############################

def compute_standard_transformation(X, Y):
    """
    Compute a 3x3 correction matrix using matrix_colour_correction,
    such that Y ≈ X @ M.
    """
    M = matrix_colour_correction(X, Y)
    return M

def compute_r_squared_standard(X, Y, M):
    """
    Compute R² for the standard method using:
         Y_pred = X @ M.T
    """
    Y_pred = np.dot(X, M.T)
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def generate_cube_standard(M, lut_size=33, exposure_gain=1.0):
    """
    Generate a 3D LUT using the standard 3x3 correction matrix.
    For each input [r, g, b], compute output = M * [r, g, b] then apply exposure_gain.
    """
    grid = np.linspace(0, 1, lut_size)
    lut = []
    for b in grid:
        for g in grid:
            for r in grid:
                rgb_in = np.array([r, g, b])
                rgb_out = np.dot(M, rgb_in) * exposure_gain
                rgb_out = np.clip(rgb_out, 0, 1)
                lut.append(rgb_out)
    return np.array(lut)

def preview_transformation_standard(image, M):
    """
    Apply the standard 3x3 matrix to an image.
    """
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3)
    transformed = np.dot(pixels, M.T)
    transformed = np.clip(transformed, 0, 1)
    return transformed.reshape(h, w, 3)

###############################
# Robust Transformation
###############################

def compute_robust_transformation(X, Y):
    """
    Compute an affine transformation using robust regression (RANSAC).
    Fits a separate robust model for each channel.
    Returns a 3x4 matrix T such that Y ≈ [X, 1] @ T.T
    """
    T = np.zeros((3, 4))
    for i in range(3):
        ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=0.5, residual_threshold=0.05)
        ransac.fit(X, Y[:, i])
        coef = ransac.estimator_.coef_
        intercept = ransac.estimator_.intercept_
        T[i, :3] = coef
        T[i, 3] = intercept
    return T

def compute_r_squared_robust(X, Y, T):
    """
    Compute R² for the robust method.
    Augment X with ones: X_aug = [X, ones], then Y_pred = X_aug @ T.T.
    """
    n = X.shape[0]
    X_aug = np.hstack([X, np.ones((n, 1))])
    Y_pred = np.dot(X_aug, T.T)
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def generate_cube_robust(T, lut_size=33, exposure_gain=1.0):
    """
    Generate a 3D LUT using the robust affine transformation.
    For each input RGB, form [r, g, b, 1] and compute output = [r, g, b, 1] @ T.T,
    then apply the exposure_gain.
    """
    grid = np.linspace(0, 1, lut_size)
    lut = []
    for b in grid:
        for g in grid:
            for r in grid:
                rgb_in = np.array([r, g, b, 1.0])
                rgb_out = np.dot(rgb_in, T.T) * exposure_gain
                rgb_out = np.clip(rgb_out, 0, 1)
                lut.append(rgb_out)
    return np.array(lut)

def preview_transformation_robust(image, T):
    """
    Apply the robust affine transformation to an image.
    For each pixel, form [r, g, b, 1] and compute output = [r, g, b, 1] @ T.T.
    """
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3)
    pixels_aug = np.hstack([pixels, np.ones((pixels.shape[0], 1))])
    transformed = np.dot(pixels_aug, T.T)
    transformed = np.clip(transformed, 0, 1)
    return transformed.reshape(h, w, 3)

###############################
# Main Routine
###############################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} <base_folder> [--robust]".format(sys.argv[0]))
        print("  The base folder should contain subfolders 'nikon' and 'hasselblad'.")
        print("  Include --robust to use the robust (affine) method.")
        sys.exit(1)
    
    base_folder = sys.argv[1]
    use_robust = False
    if len(sys.argv) >= 3 and sys.argv[2] == "--robust":
        use_robust = True

    nikon_folder = os.path.join(base_folder, "nikon")
    hasselblad_folder = os.path.join(base_folder, "hasselblad")
    
    nikon_files = get_image_files(nikon_folder)
    hasselblad_files = get_image_files(hasselblad_folder)
    
    print("Nikon files found:", len(nikon_files))
    print("Hasselblad files found:", len(hasselblad_files))
    
    if len(nikon_files) != len(hasselblad_files):
        raise Exception("The number of Nikon and Hasselblad images must match.")
    
    # Optionally enable visualization of detected patches.
    visualize_detection = True
    X, Y = gather_patch_data(nikon_files, hasselblad_files, visualize_detection)
    
    if not use_robust:
        print("\nUsing STANDARD transformation method.")
        M = compute_standard_transformation(X, Y)
        print("Computed 3x3 Correction Matrix (Nikon -> Hasselblad):\n", M)
        r2 = compute_r_squared_standard(X, Y, M)
        print("R² of the standard correction: {:.4f}".format(r2))
        auto_gain = compute_automatic_exposure_gain(X, Y, M, robust=False)
        print("Automatic exposure gain factor: {:.4f}".format(auto_gain))
        lut = generate_cube_standard(M, lut_size=33, exposure_gain=auto_gain)
        test_image = read_image(nikon_files[0])
        transformed_image = preview_transformation_standard(test_image, M)
    else:
        print("\nUsing ROBUST transformation method.")
        T = compute_robust_transformation(X, Y)
        print("Computed Robust Affine Transformation Matrix (3x4):\n", T)
        r2 = compute_r_squared_robust(X, Y, T)
        print("R² of the robust correction: {:.4f}".format(r2))
        auto_gain = compute_automatic_exposure_gain(X, Y, T, robust=True)
        print("Automatic exposure gain factor: {:.4f}".format(auto_gain))
        lut = generate_cube_robust(T, lut_size=33, exposure_gain=auto_gain)
        test_image = read_image(nikon_files[0])
        transformed_image = preview_transformation_robust(test_image, T)
    
    # Write LUT to file.
    output_filename = os.path.join(base_folder, "nikon_to_hasselblad.cube")
    with open(output_filename, "w") as f:
        f.write("TITLE \"Nikon to Hasselblad LUT\"\n")
        f.write("LUT_3D_SIZE 33\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
        for entry in lut:
            f.write("{:.6f} {:.6f} {:.6f}\n".format(entry[0], entry[1], entry[2]))
    print("Cube LUT written to:", output_filename)
    
    # Preview transformation on the first Nikon image.
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(test_image)
    ax[0].set_title("Original Nikon Image")
    ax[0].axis("off")
    ax[1].imshow(transformed_image)
    if not use_robust:
        ax[1].set_title("Transformed Image (Standard Matrix Applied)")
    else:
        ax[1].set_title("Transformed Image (Robust Affine Applied)")
    ax[1].axis("off")
    plt.tight_layout()
    plt.show()
