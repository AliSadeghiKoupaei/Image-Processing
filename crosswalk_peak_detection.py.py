import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load the image
image_path = r'C:\Users\usaal\OneDrive\Desktop\satellite_image.png'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Detection information (Rotated Bounding Box)
obb = {
    'x_center': 215.01454162597656,
    'y_center': 427.5213928222656,
    'width': 175.68026733398438,
    'height': 59.068443298339844,
    'rotation': 0.5420379638671875
}

# Perspective transformation
angle = np.degrees(obb['rotation'])
center = (int(obb['x_center']), int(obb['y_center']))
size = (int(obb['width']), int(obb['height']))
rect = (center, size, angle)
box = cv2.boxPoints(rect)
box = np.int32(box)
width = int(rect[1][0])
height = int(rect[1][1])
dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
M = cv2.getPerspectiveTransform(box.astype(np.float32), dst_pts)
cropped = cv2.warpPerspective(image, M, (width, height))
cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
cropped_transposed = np.transpose(cropped, (1, 0, 2))

# Display image
plt.figure(figsize=(6, 6))
plt.imshow(cropped_transposed)
plt.title("Transposed Cropped Image")
plt.axis('off')
plt.tight_layout()
plt.show()

# Signal extraction
pixel_matrix = np.array(cropped_transposed)
average_values_2d = np.mean(pixel_matrix, axis=2)
average_values = average_values_2d.flatten()
pixel_indices = np.arange(len(average_values))

df = pd.DataFrame({
    'Pixel Index': pixel_indices,
    'Average RGB': average_values
})

# Normalize the signal globally
df['Normalized Average RGB'] = (df['Average RGB'] - df['Average RGB'].min()) / (df['Average RGB'].max() - df['Average RGB'].min())
y_values = df['Normalized Average RGB'].values

# Dot plot
plt.figure(figsize=(12, 4))
plt.scatter(df['Pixel Index'], y_values, color='gray', s=1)
plt.title('Pixel Index vs Normalized Average RGB (Dot Plot)')
plt.xlabel('Pixel Index')
plt.ylabel('Normalized Average RGB')
plt.tight_layout()
plt.show()

# Fourier Transform
fft_values = np.fft.fft(y_values)
frequencies = np.fft.fftfreq(len(y_values))
threshold = 0.001
fft_values_filtered = np.where(np.abs(frequencies) > threshold, 0, fft_values)
smoothed_values = np.fft.ifft(fft_values_filtered).real

# Peak Detection
peaks, _ = find_peaks(smoothed_values, distance=500)
amplitudes = []
periods = []

for i, peak_idx in enumerate(peaks):
    left_bound = max(0, peak_idx - 200)
    right_bound = min(len(smoothed_values), peak_idx + 200)
    local_min = np.min(smoothed_values[left_bound:right_bound])
    local_max = smoothed_values[peak_idx]
    amplitude = local_max - local_min
    amplitudes.append(amplitude)
    if i > 0:
        period = peaks[i] - peaks[i - 1]
        periods.append(period)

print(f"\nDetected {len(peaks)} peaks.")
for i, peak_idx in enumerate(peaks):
    print(f"Peak {i+1} at Pixel {peak_idx} -> Amplitude: {amplitudes[i]:.3f}")

if periods:
    print(f"Estimated average period: {np.mean(periods):.2f} pixels\n")

# Plot with peaks marked
plt.figure(figsize=(12, 4))
plt.plot(df['Pixel Index'], smoothed_values, color='blue', label='Smoothed Signal')
plt.scatter(peaks, smoothed_values[peaks], color='red', label='Detected Peaks')
for i, peak_idx in enumerate(peaks):
    plt.text(peak_idx, smoothed_values[peak_idx], f'Peak {i+1}', fontsize=8, ha='center')
plt.title('Extracted Pattern with Peaks')
plt.xlabel('Pixel Index')
plt.ylabel('Smoothed Signal')
plt.legend()
plt.tight_layout()
plt.show()

# Plot with vertical lines at every x=1000
plt.figure(figsize=(12, 4))
plt.plot(df['Pixel Index'], y_values, color='gray', alpha=0.5, label='Noisy Signal')
plt.plot(df['Pixel Index'], smoothed_values, color='blue', label='Extracted Pattern')
for x in range(1000, len(df['Pixel Index']), 1000):
    plt.axvline(x=x, color='red', linestyle='--', linewidth=0.8)
plt.title('Noisy Signal with Vertical Lines')
plt.xlabel('Pixel Index')
plt.ylabel('Normalized Average RGB')
plt.legend()
plt.tight_layout()
plt.show()

# Plot with both x=100 and x=1000 vertical lines + annotations
plt.figure(figsize=(12, 4))
plt.plot(df['Pixel Index'], y_values, color='gray', alpha=0.5, label='Noisy Signal')
plt.plot(df['Pixel Index'], smoothed_values, color='blue', label='Extracted Pattern')

for x in range(100, len(df['Pixel Index']), 100):
    plt.axvline(x=x, color='red', linestyle='--', linewidth=0.5)

for x in range(1000, len(df['Pixel Index']), 1000):
    plt.axvline(x=x, color='red', linestyle='-', linewidth=0.8)
    y_value = smoothed_values[x]
    plt.scatter(x, y_value, color='red')
    plt.text(x, y_value, f'{y_value:.2f}', color='red', ha='right', fontsize=8, rotation=45)

plt.title('Annotated Signal')
plt.xlabel('Pixel Index')
plt.ylabel('Normalized Average RGB')
plt.legend()
plt.tight_layout()
plt.show()