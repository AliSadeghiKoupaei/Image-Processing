import cv2
import numpy as np

input_path = "C:\\Users\\usaal\\OneDrive\\Desktop\\images\\test-img6.pgm"
output_path = "C:\\Users\\usaal\\OneDrive\\Desktop\\test-img6-local.pgm"

window_size = 25
k0 = 0.8
k1 = 0.001
k2 = 0.8
E = 7.0

img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found at {input_path}")

f = img.astype(np.float32)

mG = f.mean()
sigmaG = f.std()

kernel = (window_size, window_size)
local_mean = cv2.boxFilter(f, ddepth=-1, ksize=kernel, normalize=True)
local_sq_mean = cv2.boxFilter(f**2, ddepth=-1, ksize=kernel, normalize=True)
local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))

mask = (local_mean < k0 * mG) & (local_std >= k1 * sigmaG) & (local_std <= k2 * sigmaG)

g = f.copy()
g[mask] = E * g[mask]

g = np.clip(g, 0, 255).astype(np.uint8)

cv2.imwrite(output_path, g)
print(f"Enhanced image saved to: {output_path}")