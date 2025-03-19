# Color-Match-LUT
Take shots of color checker and create LUT based on pairs of tif image to match color of two cameras. I used this to match my Nikon Z6iii to Hasselbald 907x 50c.

## Usage

```python
python generate.py <base_folder_path> <method_flag> <display_patch>
```

The base folder must contain two subfolders, "nikon" and "hasselblad", each containing calibration images (.tif and/or .TIF).

For method_flag, use one of the following flags (default is --standard):
    
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
    --compare    : Compare all methods and view their R² values, exposure gains, and previews.              
                   
For display_patch, use the following flag (no flag means display patches):
  
    --no-display-patches : Do not display the color patch images (color checker detection).
                           Only the final result will be shown.

## Example Display Patch

![Figure_2](https://github.com/user-attachments/assets/5442a1df-013f-446e-b8ec-02a03eda8ba3)

## Example Result

In my case, LOESS takes a huge amount of time (3 hours in my case), but it is the most accurate.

![Figure_1](https://github.com/user-attachments/assets/08e0de3e-8d82-4d54-88b5-5d14f939a74f)

## Development

I used mostly ChatGPT for this, as I have no idea how color works. So feel free to modify anything that is incorrect.
