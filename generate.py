#!/usr/bin/env python3
"""
Usage:
  To run a single method, use one of the following flags (default is --standard):
    --standard   : Global linear transformation using Colour's matrix_colour_correction.
                   (Computes a 3x3 matrix mapping Nikon patches to Hasselblad patches.)
    --robust     : Robust affine transformation using RANSAC (3x4 matrix with intercepts).
                   (Fits a separate linear model per channel and is robust to outliers.)
    --nonlinear  : Nonlinear (polynomial) transformation using a degree-2 polynomial.
                   (Captures nonlinearities in the color mapping.)
    --piecewise  : Piecewise linear transformation that segments the data by luminance
                   (shadows, midtones, highlights) and interpolates between separate matrices.
    --mlp        : Neural network mapping using an MLPRegressor.
                   (A simple feedforward network learns the mapping.)
    --iterative  : Iterative refinement of the standard matrix using gradient descent.
                   (Starts from the standard solution and refines it.)
    --kernel     : Kernel regression (Nadaraya–Watson) with a Gaussian kernel.
                   (Predicts outputs as a weighted average of calibration points.)
    --loess      : LOESS (locally weighted linear regression) for local models.
                   (Fits a local linear model for each new input using Gaussian weighting.)
    --local      : Local linear models using clustering (e.g. k-means).
                   (Clusters the data and fits separate linear models in each cluster.)
    --ridge      : Regularized regression (Ridge) for a global affine mapping.
                   (Uses L2 regularization for stability.)
                   
  Additionally, add the flag:
    --no-display-patches : Do not display the color patch images (color checker detection).
                           Only the corrected result will be shown.
                           
  To compare all methods and view their R² values, exposure gains, and previews, use:
    --compare

The base folder must contain two subfolders, "nikon" and "hasselblad", each containing calibration images (.tif and/or .TIF).
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

from colour import read_image, matrix_colour_correction
from colour_checker_detection import detect_colour_checkers_segmentation

# For robust & regularized regression
from sklearn.linear_model import RANSACRegressor, LinearRegression, Ridge
# For nonlinear regression (polynomial & MLP)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
# For clustering (local linear)
from sklearn.cluster import KMeans

###############################
# Utility Functions
###############################

def get_image_files(folder):
    files_lower = glob.glob(os.path.join(folder, "*.tif"))
    files_upper = glob.glob(os.path.join(folder, "*.TIF"))
    return sorted(files_lower + files_upper)

def reshape_patches(patch_values):
    if patch_values.shape[0] == 24:
        return patch_values.reshape(4,6,3)
    else:
        return patch_values.reshape(-1,1,3)

def visualize_checker_detection_both(nikon_img, nikon_patches, nikon_file,
                                     hasselblad_img, hasselblad_patches, hasselblad_file):
    np_grid = reshape_patches(nikon_patches)
    hp_grid = reshape_patches(hasselblad_patches)
    fig, axes = plt.subplots(2,2, figsize=(12,8))
    axes[0,0].imshow(nikon_img)
    axes[0,0].set_title(f"Nikon: {os.path.basename(nikon_file)}")
    axes[0,0].axis("off")
    axes[0,1].imshow(hasselblad_img)
    axes[0,1].set_title(f"Hasselblad: {os.path.basename(hasselblad_file)}")
    axes[0,1].axis("off")
    axes[1,0].imshow(np_grid)
    axes[1,0].set_title("Nikon Patches")
    axes[1,0].axis("off")
    axes[1,1].imshow(hp_grid)
    axes[1,1].set_title("Hasselblad Patches\n(Exposure Corrected)")
    axes[1,1].axis("off")
    plt.tight_layout()
    plt.show()

def calibrate_exposure_using_bottom_row(nikon_patches, hasselblad_patches):
    if nikon_patches.shape[0]==24 and hasselblad_patches.shape[0]==24:
        np_grid = nikon_patches.reshape(4,6,3)
        hp_grid = hasselblad_patches.reshape(4,6,3)
        nikon_mean = np.mean(np_grid[3,:,:])
        hasselblad_mean = np.mean(hp_grid[3,:,:])
    else:
        nikon_mean = np.mean(nikon_patches[np.argmin(np.std(nikon_patches, axis=1))])
        hasselblad_mean = np.mean(hasselblad_patches[np.argmin(np.std(hasselblad_patches, axis=1))])
    return 1.0 if hasselblad_mean==0 else nikon_mean/hasselblad_mean

def gather_patch_data(nikon_files, hasselblad_files, visualize_detection=False):
    sources, targets = [], []
    for nfile, hfile in zip(nikon_files, hasselblad_files):
        n_img = read_image(nfile)
        h_img = read_image(hfile)
        n_check = detect_colour_checkers_segmentation(n_img)
        h_check = detect_colour_checkers_segmentation(h_img)
        if len(n_check)!=1 or len(h_check)!=1:
            print(f"Skipping {nfile} / {hfile} due to detection issues.")
            continue
        n_patches = np.array(n_check[0])
        h_patches = np.array(h_check[0])
        if n_patches.shape != h_patches.shape:
            print(f"Skipping {nfile} / {hfile} due to patch count mismatch.")
            continue
        gain = calibrate_exposure_using_bottom_row(n_patches, h_patches)
        h_patches = np.clip(h_patches*gain, 0, 1)
        if visualize_detection:
            visualize_checker_detection_both(n_img, n_patches, nfile, h_img, h_patches, hfile)
        sources.append(n_patches)
        targets.append(h_patches)
    if not sources:
        raise Exception("No valid calibration pairs found!")
    X = np.vstack(sources)
    Y = np.vstack(targets)
    return X, Y

def preview_transformation(image, predict_func):
    h, w, _ = image.shape
    pixels = image.reshape(-1,3)
    transformed = predict_func(pixels)
    return np.clip(transformed, 0, 1).reshape(h,w,3)

def compute_exposure_gain(predict_func, model, X, Y):
    Y_pred = predict_func(model, X)
    m_target, m_pred = np.mean(Y), np.mean(Y_pred)
    return 1.0 if m_pred==0 else m_target/m_pred

###############################
# Transformation Methods
###############################

def compute_standard_transformation(X, Y):
    return matrix_colour_correction(X, Y)

def predict_standard(M, X_new):
    return np.dot(X_new, M.T)

def generate_cube_standard(M, lut_size=33, exposure_gain=1.0):
    grid = np.linspace(0,1, lut_size)
    lut = []
    for b in grid:
        for g in grid:
            for r in grid:
                rgb_in = np.array([r, g, b])
                rgb_out = np.dot(M, rgb_in)*exposure_gain
                lut.append(np.clip(rgb_out, 0, 1))
    return np.array(lut)

def preview_transformation_standard(image, M):
    h, w, _ = image.shape
    pixels = image.reshape(-1,3)
    transformed = np.dot(pixels, M.T)
    return np.clip(transformed, 0, 1).reshape(h,w,3)

def compute_robust_transformation(X, Y):
    T = np.zeros((3,4))
    for i in range(3):
        ransac = RANSACRegressor(LinearRegression(), min_samples=0.5, residual_threshold=0.05)
        ransac.fit(X, Y[:,i])
        T[i, :3] = ransac.estimator_.coef_
        T[i, 3] = ransac.estimator_.intercept_
    return T

def predict_robust(T, X_new):
    n = X_new.shape[0]
    X_aug = np.hstack([X_new, np.ones((n,1))])
    return np.dot(X_aug, T.T)

def generate_cube_robust(T, lut_size=33, exposure_gain=1.0):
    grid = np.linspace(0,1, lut_size)
    lut = []
    for b in grid:
        for g in grid:
            for r in grid:
                rgb_in = np.array([r, g, b, 1.0])
                rgb_out = np.dot(rgb_in, T.T)*exposure_gain
                lut.append(np.clip(rgb_out, 0,1))
    return np.array(lut)

def preview_transformation_robust(image, T):
    h, w, _ = image.shape
    pixels = image.reshape(-1,3)
    X_aug = np.hstack([pixels, np.ones((pixels.shape[0],1))])
    transformed = np.dot(X_aug, T.T)
    return np.clip(transformed, 0,1).reshape(h,w,3)

def compute_nonlinear_transformation(X, Y, degree=2):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, Y)
    return model

def predict_nonlinear(model, X_new):
    return model.predict(X_new)

def generate_cube_nonlinear(model, lut_size=33, exposure_gain=1.0):
    grid = np.linspace(0,1, lut_size)
    lut = []
    for b in grid:
        for g in grid:
            for r in grid:
                rgb_in = np.array([r, g, b]).reshape(1,-1)
                rgb_out = model.predict(rgb_in)[0]*exposure_gain
                lut.append(np.clip(rgb_out,0,1))
    return np.array(lut)

def preview_transformation_nonlinear(image, model):
    h, w, _ = image.shape
    pixels = image.reshape(-1,3)
    transformed = model.predict(pixels)
    return np.clip(transformed,0,1).reshape(h,w,3)

def compute_piecewise_transformation(X, Y):
    L = 0.2126*X[:,0] + 0.7152*X[:,1] + 0.0722*X[:,2]
    t_low = np.percentile(L,33)
    t_mid = np.percentile(L,50)
    t_high = np.percentile(L,66)
    shadows = np.where(L<=t_low)[0]
    mids = np.where((L>t_low)&(L<t_high))[0]
    highs = np.where(L>=t_high)[0]
    M_shadow = matrix_colour_correction(X[shadows], Y[shadows]) if shadows.size>=3 else matrix_colour_correction(X,Y)
    M_mid = matrix_colour_correction(X[mids], Y[mids]) if mids.size>=3 else matrix_colour_correction(X,Y)
    M_high = matrix_colour_correction(X[highs], Y[highs]) if highs.size>=3 else matrix_colour_correction(X,Y)
    return t_low, t_mid, t_high, M_shadow, M_mid, M_high

def predict_piecewise(t_low, t_mid, t_high, M_shadow, M_mid, M_high, X_new):
    N = X_new.shape[0]
    Y_pred = np.zeros_like(X_new)
    for i in range(N):
        rgb = X_new[i]
        L_val = 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]
        if L_val<=t_low:
            Y_pred[i] = np.dot(M_shadow, rgb)
        elif L_val>=t_high:
            Y_pred[i] = np.dot(M_high, rgb)
        elif L_val < t_mid:
            w = (L_val-t_low)/(t_mid-t_low)
            M_interp = (1-w)*M_shadow + w*M_mid
            Y_pred[i] = np.dot(M_interp, rgb)
        else:
            w = (L_val-t_mid)/(t_high-t_mid)
            M_interp = (1-w)*M_mid + w*M_high
            Y_pred[i] = np.dot(M_interp, rgb)
    return Y_pred

def generate_cube_piecewise(t_low, t_mid, t_high, M_shadow, M_mid, M_high, lut_size=33, exposure_gain=1.0):
    grid = np.linspace(0,1, lut_size)
    lut = []
    for b in grid:
        for g in grid:
            for r in grid:
                rgb_in = np.array([r, g, b])
                L_val = 0.2126*r + 0.7152*g + 0.0722*b
                if L_val<=t_low:
                    M_interp = M_shadow
                elif L_val>=t_high:
                    M_interp = M_high
                elif L_val < t_mid:
                    w = (L_val-t_low)/(t_mid-t_low)
                    M_interp = (1-w)*M_shadow + w*M_mid
                else:
                    w = (L_val-t_mid)/(t_high-t_mid)
                    M_interp = (1-w)*M_mid + w*M_high
                rgb_out = np.dot(M_interp, rgb_in)*exposure_gain
                lut.append(np.clip(rgb_out,0,1))
    return np.array(lut)

def preview_transformation_piecewise(image, t_low, t_mid, t_high, M_shadow, M_mid, M_high):
    h, w, _ = image.shape
    pixels = image.reshape(-1,3)
    transformed = np.zeros_like(pixels)
    for i in range(pixels.shape[0]):
        rgb = pixels[i]
        L_val = 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]
        if L_val<=t_low:
            M_interp = M_shadow
        elif L_val>=t_high:
            M_interp = M_high
        elif L_val < t_mid:
            wght = (L_val-t_low)/(t_mid-t_low)
            M_interp = (1-wght)*M_shadow + wght*M_mid
        else:
            wght = (L_val-t_mid)/(t_high-t_mid)
            M_interp = (1-wght)*M_mid + wght*M_high
        transformed[i] = np.dot(M_interp, rgb)
    return np.clip(transformed, 0,1).reshape(h,w,3)

def compute_mlp_transformation(X, Y):
    mlp = MLPRegressor(hidden_layer_sizes=(20,20), max_iter=10000, random_state=42)
    mlp.fit(X, Y)
    return mlp

def predict_mlp(model, X_new):
    return model.predict(X_new)

def generate_cube_mlp(mlp, lut_size=33, exposure_gain=1.0):
    grid = np.linspace(0,1, lut_size)
    lut = []
    for b in grid:
        for g in grid:
            for r in grid:
                rgb_in = np.array([r, g, b]).reshape(1,-1)
                rgb_out = mlp.predict(rgb_in)[0]*exposure_gain
                lut.append(np.clip(rgb_out, 0,1))
    return np.array(lut)

def preview_transformation_mlp(image, mlp):
    h, w, _ = image.shape
    pixels = image.reshape(-1,3)
    transformed = mlp.predict(pixels)
    return np.clip(transformed,0,1).reshape(h,w,3)

def compute_iterative_transformation(X, Y, lr=0.001, iterations=1000):
    M = compute_standard_transformation(X, Y)
    n = X.shape[0]
    for i in range(iterations):
        Y_pred = np.dot(X, M.T)
        error = Y - Y_pred
        grad = -2*np.dot(error.T, X)/n
        M = M - lr*grad.T
    return M

def predict_iterative(M, X_new):
    return np.dot(X_new, M.T)

def generate_cube_iterative(M, lut_size=33, exposure_gain=1.0):
    grid = np.linspace(0,1, lut_size)
    lut = []
    for b in grid:
        for g in grid:
            for r in grid:
                rgb_in = np.array([r, g, b])
                rgb_out = np.dot(M, rgb_in)*exposure_gain
                lut.append(np.clip(rgb_out,0,1))
    return np.array(lut)

def preview_transformation_iterative(image, M):
    return preview_transformation_standard(image, M)

def compute_kernel_transformation(X, Y, sigma=0.1):
    return (X, Y, sigma)

def predict_kernel(model, X_new):
    X_train, Y_train, sigma = model
    N = X_new.shape[0]
    Y_pred = np.zeros((N, Y_train.shape[1]))
    for i in range(N):
        diff = X_train - X_new[i]
        d2 = np.sum(diff**2, axis=1)
        weights = np.exp(-d2/(2*sigma**2))
        if np.sum(weights)==0:
            Y_pred[i] = np.zeros(Y_train.shape[1])
        else:
            Y_pred[i] = np.sum(weights[:,None]*Y_train, axis=0)/np.sum(weights)
    return Y_pred

def compute_loess_transformation(X, Y, sigma=0.1):
    X_aug = np.hstack([X, np.ones((X.shape[0],1))])
    return (X_aug, Y, sigma)

def predict_loess(model, X_new):
    X_aug_train, Y_train, sigma = model
    N = X_new.shape[0]
    Y_pred = np.zeros((N, Y_train.shape[1]))
    for i in range(N):
        x_new_aug = np.hstack([X_new[i], 1])
        diff = X_aug_train[:,:-1] - X_new[i]
        d2 = np.sum(diff**2, axis=1)
        weights = np.exp(-d2/(2*sigma**2))
        W = np.diag(weights)
        try:
            beta = np.linalg.inv(X_aug_train.T @ W @ X_aug_train) @ (X_aug_train.T @ W @ Y_train)
        except np.linalg.LinAlgError:
            beta = np.zeros((X_aug_train.shape[1], Y_train.shape[1]))
        Y_pred[i] = np.dot(x_new_aug, beta)
    return Y_pred

def compute_local_linear_transformation(X, Y, n_clusters=3):
    X = X.astype(np.float64)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    models = {}
    for k in range(n_clusters):
        idx = np.where(labels==k)[0]
        if idx.size < 3:
            models[k] = LinearRegression().fit(X, Y)
        else:
            models[k] = LinearRegression().fit(X[idx], Y[idx])
    return (kmeans, models)

def predict_local_linear(model, X_new):
    kmeans, models = model
    X_new = X_new.astype(np.float64)
    labels = kmeans.predict(X_new)
    Y_pred = np.zeros((X_new.shape[0], Y.shape[1]))
    for k, reg in models.items():
        idx = np.where(labels==k)[0]
        if idx.size > 0:
            Y_pred[idx] = reg.predict(X_new[idx])
    return Y_pred

def compute_ridge_transformation(X, Y, alpha=1.0):
    T = np.zeros((3,4))
    for i in range(3):
        ridge = Ridge(alpha=alpha)
        ridge.fit(X, Y[:,i])
        T[i,:3] = ridge.coef_
        T[i,3] = ridge.intercept_
    return T

def predict_ridge(T, X_new):
    n = X_new.shape[0]
    X_aug = np.hstack([X_new, np.ones((n,1))])
    return np.dot(X_aug, T.T)

def generate_cube(predict_func, model, lut_size=33, exposure_gain=1.0):
    grid = np.linspace(0,1, lut_size)
    lut = []
    for b in grid:
        for g in grid:
            for r in grid:
                rgb_in = np.array([r, g, b]).reshape(1,-1)
                rgb_out = predict_func(model, rgb_in)[0]*exposure_gain
                lut.append(np.clip(rgb_out,0,1))
    return np.array(lut)

###############################
# Global r2 Function (moved to global scope)
###############################

def compute_r2(Y_true, Y_pred):
    ss_res = np.sum((Y_true - Y_pred)**2)
    ss_tot = np.sum((Y_true - np.mean(Y_true, axis=0))**2)
    return 1 - ss_res/ss_tot

###############################
# Multiprocessing worker for compare mode
###############################

def run_method(meth):
    # Uses globals: X, Y, test_image, compute_r2
    if meth=="standard":
        M = compute_standard_transformation(X, Y)
        Y_pred = np.dot(X, M.T)
        r2_val = compute_r2(Y, Y_pred)
        gain_val = compute_exposure_gain(lambda m, Xnew: np.dot(Xnew, m.T), M, X, Y)
        lut_data = generate_cube(lambda m, Xnew: np.dot(Xnew, m.T), M, exposure_gain=gain_val)
        preview_img = preview_transformation_standard(test_image, M)
        title = "Standard"
    elif meth=="robust":
        T = compute_robust_transformation(X, Y)
        Y_pred = predict_robust(T, X)
        r2_val = compute_r2(Y, Y_pred)
        gain_val = compute_exposure_gain(lambda m, Xnew: predict_robust(m, Xnew), T, X, Y)
        lut_data = generate_cube(lambda m, Xnew: predict_robust(m, Xnew), T, exposure_gain=gain_val)
        preview_img = preview_transformation_robust(test_image, T)
        title = "Robust"
    elif meth=="nonlinear":
        model = compute_nonlinear_transformation(X, Y, degree=2)
        Y_pred = predict_nonlinear(model, X)
        r2_val = compute_r2(Y, Y_pred)
        gain_val = compute_exposure_gain(lambda m, Xnew: m.predict(Xnew), model, X, Y)
        lut_data = generate_cube(lambda m, Xnew: m.predict(Xnew), model, exposure_gain=gain_val)
        preview_img = preview_transformation_nonlinear(test_image, model)
        title = "Nonlinear"
    elif meth=="piecewise":
        t_low, t_mid, t_high, M_shadow, M_mid, M_high = compute_piecewise_transformation(X, Y)
        Y_pred = predict_piecewise(t_low, t_mid, t_high, M_shadow, M_mid, M_high, X)
        r2_val = compute_r2(Y, Y_pred)
        gain_val = compute_exposure_gain(lambda m, Xnew: predict_piecewise(t_low, t_mid, t_high, M_shadow, M_mid, M_high, Xnew), None, X, Y)
        lut_data = generate_cube(lambda m, Xnew: predict_piecewise(t_low, t_mid, t_high, M_shadow, M_mid, M_high, Xnew), None, exposure_gain=gain_val)
        preview_img = preview_transformation_piecewise(test_image, t_low, t_mid, t_high, M_shadow, M_mid, M_high)
        title = "Piecewise"
    elif meth=="mlp":
        mlp_model = compute_mlp_transformation(X, Y)
        Y_pred = predict_mlp(mlp_model, X)
        r2_val = compute_r2(Y, Y_pred)
        gain_val = compute_exposure_gain(lambda m, Xnew: m.predict(Xnew), mlp_model, X, Y)
        lut_data = generate_cube(lambda m, Xnew: m.predict(Xnew), mlp_model, exposure_gain=gain_val)
        preview_img = preview_transformation_mlp(test_image, mlp_model)
        title = "MLP"
    elif meth=="iterative":
        M_iter = compute_iterative_transformation(X, Y, lr=0.001, iterations=1000)
        Y_pred = np.dot(X, M_iter.T)
        r2_val = compute_r2(Y, Y_pred)
        gain_val = compute_exposure_gain(lambda m, Xnew: np.dot(Xnew, m.T), M_iter, X, Y)
        lut_data = generate_cube(lambda m, Xnew: np.dot(Xnew, m.T), M_iter, exposure_gain=gain_val)
        preview_img = preview_transformation_standard(test_image, M_iter)
        title = "Iterative"
    elif meth=="kernel":
        ker_model = compute_kernel_transformation(X, Y, sigma=0.1)
        Y_pred = predict_kernel(ker_model, X)
        r2_val = compute_r2(Y, Y_pred)
        gain_val = compute_exposure_gain(lambda m, Xnew: predict_kernel(m, Xnew), ker_model, X, Y)
        lut_data = generate_cube(lambda m, Xnew: predict_kernel(m, Xnew), ker_model, exposure_gain=gain_val)
        preview_img = preview_transformation(test_image, lambda x: predict_kernel(ker_model, x))
        title = "Kernel"
    elif meth=="loess":
        loess_model = compute_loess_transformation(X, Y, sigma=0.1)
        Y_pred = predict_loess(loess_model, X)
        r2_val = compute_r2(Y, Y_pred)
        gain_val = compute_exposure_gain(lambda m, Xnew: predict_loess(m, Xnew), loess_model, X, Y)
        lut_data = generate_cube(lambda m, Xnew: predict_loess(m, Xnew), loess_model, exposure_gain=gain_val)
        preview_img = preview_transformation(test_image, lambda x: predict_loess(loess_model, x))
        title = "LOESS"
    elif meth=="local":
        local_model = compute_local_linear_transformation(X, Y, n_clusters=3)
        Y_pred = predict_local_linear(local_model, X)
        r2_val = compute_r2(Y, Y_pred)
        gain_val = 1.0
        lut_data = generate_cube(lambda m, Xnew: predict_local_linear(m, Xnew), local_model, exposure_gain=gain_val)
        preview_img = preview_transformation(test_image, lambda x: predict_local_linear(local_model, x))
        title = "Local"
    elif meth=="ridge":
        ridge_T = compute_ridge_transformation(X, Y, alpha=1.0)
        Y_pred = predict_ridge(ridge_T, X)
        r2_val = compute_r2(Y, Y_pred)
        gain_val = compute_exposure_gain(lambda m, Xnew: predict_ridge(m, Xnew), ridge_T, X, Y)
        lut_data = generate_cube(lambda m, Xnew: predict_ridge(m, Xnew), ridge_T, exposure_gain=gain_val)
        preview_img = preview_transformation(test_image, lambda x: predict_ridge(ridge_T, x))
        title = "Ridge"
    else:
        raise Exception("Unknown method flag in worker")
    return {"title": title, "r2": r2_val, "gain": gain_val, "lut": lut_data, "preview": preview_img}

###############################
# Pool initializer to share globals
###############################
def init_worker(X_, Y_, test_image_, compute_r2_):
    global X, Y, test_image, compute_r2
    X = X_
    Y = Y_
    test_image = test_image_
    compute_r2 = compute_r2_

###############################
# Main Routine
###############################
if __name__=="__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
        
    base_folder = sys.argv[1]
    method_flag = sys.argv[2].lower() if len(sys.argv) >= 3 and sys.argv[2].startswith("--") else "--standard"
    method = method_flag.replace("--", "")
    
    visualize_detection = True
    if "--no-display-patches" in sys.argv:
        visualize_detection = False

    nikon_folder = os.path.join(base_folder, "nikon")
    hasselblad_folder = os.path.join(base_folder, "hasselblad")
    nikon_files = get_image_files(nikon_folder)
    hasselblad_files = get_image_files(hasselblad_folder)
    if len(nikon_files) != len(hasselblad_files):
        raise Exception("Number of Nikon and Hasselblad images must match.")
        
    X, Y = gather_patch_data(nikon_files, hasselblad_files, visualize_detection)
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    test_image = read_image(nikon_files[0])
    
    if method == "compare":
        # Exclude spline and gpr from the comparison
        method_list = ["standard", "robust", "nonlinear", "piecewise", "mlp", "iterative",
                       "kernel", "loess", "local", "ridge"]
        # Initialize worker globals via initializer
        with Pool(initializer=init_worker, initargs=(X, Y, test_image, compute_r2)) as pool:
            results_list = pool.map(run_method, method_list)
        # Organize results by title.
        results_dict = {res["title"]: (res["r2"], res["gain"]) for res in results_list}
        luts = {res["title"]: res["lut"] for res in results_list}
        previews = {res["title"]: res["preview"] for res in results_list}
        # Write LUT files.
        for title, (r2_val, gain_val) in results_dict.items():
            out_fn = os.path.join(base_folder, f"nikon_to_hasselblad_{title.lower()}.cube")
            with open(out_fn, "w") as f:
                f.write(f"TITLE \"Nikon to Hasselblad LUT ({title})\"\n")
                f.write("LUT_3D_SIZE 33\n")
                f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
                f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
                for entry in luts[title]:
                    f.write("{:.6f} {:.6f} {:.6f}\n".format(entry[0], entry[1], entry[2]))
            print(f"LUT for {title} written to: {out_fn}")
        print("\nComparison Summary:")
        for title, (r2_val, gain_val) in results_dict.items():
            print(f"  {title}: R² = {r2_val:.4f}, Exposure Gain = {gain_val:.4f}")
        # Display preview images in a row.
        n_methods = len(results_dict)
        fig, ax = plt.subplots(1, n_methods, figsize=(4*n_methods, 6))
        for i, title in enumerate(results_dict.keys()):
            r2_val, gain_val = results_dict[title]
            ax[i].imshow(previews[title])
            ax[i].set_title(f"{title}\nR²={r2_val:.4f}\nGain={gain_val:.4f}")
            ax[i].axis("off")
        plt.tight_layout()
        plt.show()
    else:
        # Single method mode (non-compare) remains sequential.
        if method=="standard":
            M = compute_standard_transformation(X, Y)
            r2 = compute_r2(Y, np.dot(X, M.T))
            gain = compute_exposure_gain(lambda m, Xnew: np.dot(Xnew, m.T), M, X, Y)
            lut_data = generate_cube(lambda m, Xnew: np.dot(Xnew, m.T), M, exposure_gain=gain)
            preview_img = preview_transformation_standard(test_image, M)
            title = "Standard"
        elif method=="robust":
            T = compute_robust_transformation(X, Y)
            r2 = compute_r2(Y, predict_robust(T, X))
            gain = compute_exposure_gain(lambda m, Xnew: predict_robust(m, Xnew), T, X, Y)
            lut_data = generate_cube(lambda m, Xnew: predict_robust(m, Xnew), T, exposure_gain=gain)
            preview_img = preview_transformation_robust(test_image, T)
            title = "Robust"
        elif method=="nonlinear":
            model = compute_nonlinear_transformation(X, Y, degree=2)
            r2 = compute_r2(Y, predict_nonlinear(model, X))
            gain = compute_exposure_gain(lambda m, Xnew: m.predict(Xnew), model, X, Y)
            lut_data = generate_cube(lambda m, Xnew: m.predict(Xnew), model, exposure_gain=gain)
            preview_img = preview_transformation_nonlinear(test_image, model)
            title = "Nonlinear"
        elif method=="piecewise":
            t_low, t_mid, t_high, M_shadow, M_mid, M_high = compute_piecewise_transformation(X, Y)
            r2 = compute_r2(Y, predict_piecewise(t_low, t_mid, t_high, M_shadow, M_mid, M_high, X))
            gain = compute_exposure_gain(lambda m, Xnew: predict_piecewise(t_low, t_mid, t_high, M_shadow, M_mid, M_high, Xnew), None, X, Y)
            lut_data = generate_cube(lambda m, Xnew: predict_piecewise(t_low, t_mid, t_high, M_shadow, M_mid, M_high, Xnew), None, exposure_gain=gain)
            preview_img = preview_transformation_piecewise(test_image, t_low, t_mid, t_high, M_shadow, M_mid, M_high)
            title = "Piecewise"
        elif method=="mlp":
            mlp_model = compute_mlp_transformation(X, Y)
            r2 = compute_r2(Y, predict_mlp(mlp_model, X))
            gain = compute_exposure_gain(lambda m, Xnew: m.predict(Xnew), mlp_model, X, Y)
            lut_data = generate_cube(lambda m, Xnew: m.predict(Xnew), mlp_model, exposure_gain=gain)
            preview_img = preview_transformation_mlp(test_image, mlp_model)
            title = "MLP"
        elif method=="iterative":
            M_iter = compute_iterative_transformation(X, Y, lr=0.001, iterations=1000)
            r2 = compute_r2(Y, np.dot(X, M_iter.T))
            gain = compute_exposure_gain(lambda m, Xnew: np.dot(Xnew, m.T), M_iter, X, Y)
            lut_data = generate_cube(lambda m, Xnew: np.dot(Xnew, m.T), M_iter, exposure_gain=gain)
            preview_img = preview_transformation_standard(test_image, M_iter)
            title = "Iterative"
        elif method=="kernel":
            ker_model = compute_kernel_transformation(X, Y, sigma=0.1)
            Y_pred = predict_kernel(ker_model, X)
            r2 = compute_r2(Y, Y_pred)
            gain = compute_exposure_gain(lambda m, Xnew: predict_kernel(m, Xnew), ker_model, X, Y)
            lut_data = generate_cube(lambda m, Xnew: predict_kernel(m, Xnew), ker_model, exposure_gain=gain)
            preview_img = preview_transformation(test_image, lambda x: predict_kernel(ker_model, x))
            title = "Kernel"
        elif method=="loess":
            loess_model = compute_loess_transformation(X, Y, sigma=0.1)
            Y_pred = predict_loess(loess_model, X)
            r2 = compute_r2(Y, Y_pred)
            gain = compute_exposure_gain(lambda m, Xnew: predict_loess(m, Xnew), loess_model, X, Y)
            lut_data = generate_cube(lambda m, Xnew: predict_loess(m, Xnew), loess_model, exposure_gain=gain)
            preview_img = preview_transformation(test_image, lambda x: predict_loess(loess_model, x))
            title = "LOESS"
        elif method=="local":
            local_model = compute_local_linear_transformation(X, Y, n_clusters=3)
            Y_pred = predict_local_linear(local_model, X)
            r2 = compute_r2(Y, Y_pred)
            gain = 1.0
            lut_data = generate_cube(lambda m, Xnew: predict_local_linear(m, Xnew), local_model, exposure_gain=gain)
            preview_img = preview_transformation(test_image, lambda x: predict_local_linear(local_model, x))
            title = "Local"
        elif method=="ridge":
            ridge_T = compute_ridge_transformation(X, Y, alpha=1.0)
            Y_pred = predict_ridge(ridge_T, X)
            r2 = compute_r2(Y, Y_pred)
            gain = compute_exposure_gain(lambda m, Xnew: predict_ridge(m, Xnew), ridge_T, X, Y)
            lut_data = generate_cube(lambda m, Xnew: predict_ridge(m, Xnew), ridge_T, exposure_gain=gain)
            preview_img = preview_transformation(test_image, lambda x: predict_ridge(ridge_T, x))
            title = "Ridge"
        else:
            raise Exception("Unknown method flag.")
        out_fn = os.path.join(base_folder, f"nikon_to_hasselblad_{title.lower()}.cube")
        with open(out_fn, "w") as f:
            f.write(f"TITLE \"Nikon to Hasselblad LUT ({title})\"\n")
            f.write("LUT_3D_SIZE 33\n")
            f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
            f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
            for entry in lut_data:
                f.write("{:.6f} {:.6f} {:.6f}\n".format(entry[0], entry[1], entry[2]))
        print(f"LUT written to: {out_fn}")
        fig, ax = plt.subplots(1,2, figsize=(12,6))
        ax[0].imshow(test_image)
        ax[0].set_title("Original Nikon Image")
        ax[0].axis("off")
        ax[1].imshow(preview_img)
        ax[1].set_title(f"Transformed ({title})")
        ax[1].axis("off")
        plt.tight_layout()
        plt.show()
