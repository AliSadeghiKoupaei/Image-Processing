import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the image
image_path = r'C:\Users\usaal\OneDrive\Desktop\satellite_image.png'  # Ensure this is the correct path
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Detection information (Rotated Bounding Box)
obb = {
    'x_center': 215.01454162597656,
    'y_center': 427.5213928222656,
    'width': 175.68026733398438,
    'height': 59.068443298339844,
    'rotation': 0.5420379638671875  # In radians
}

# Convert rotation from radians to degrees
angle = np.degrees(obb['rotation'])

# Get bounding box parameters (convert to int)
center = (int(obb['x_center']), int(obb['y_center']))
size = (int(obb['width']), int(obb['height']))

# Define the rotated rectangle
rect = (center, size, angle)

# Get the four corner points of the rotated bounding box
box = cv2.boxPoints(rect)
box = np.int32(box)  # Convert to integer coordinates

# Find the width and height of the rotated rectangle
width = int(rect[1][0])
height = int(rect[1][1])

# Define destination points for the transformation (upright rectangle)
dst_pts = np.array([
    [0, height-1],
    [0, 0],
    [width-1, 0],
    [width-1, height-1]
], dtype="float32")

# Compute the perspective transformation matrix
M = cv2.getPerspectiveTransform(box.astype(np.float32), dst_pts)

# Apply the perspective transformation
cropped = cv2.warpPerspective(image, M, (width, height))

# Convert to RGB (OpenCV loads in BGR format)
cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

# Transpose the Cropped Image Before Further Analysis
cropped_transposed = np.transpose(cropped, (1, 0, 2))  # Transpose the image

# Display the transposed image using matplotlib
plt.figure(figsize=(6, 6))
plt.imshow(cropped_transposed)
plt.title("Transposed Cropped Image")
plt.axis('off')
plt.tight_layout()
plt.show()

# Convert the transposed image to a NumPy array
pixel_matrix = np.array(cropped_transposed)

# Calculate the average RGB value for each pixel
average_values_2d = np.mean(pixel_matrix, axis=2)

# Flatten the 2D array to a 1D array
average_values = average_values_2d.flatten()

# Create pixel indices
pixel_indices = np.arange(len(average_values))

# Create a DataFrame for plotting
df = pd.DataFrame({
    'Pixel Index': pixel_indices,
    'Average RGB': average_values
})

# Standardization for 'Average RGB' column
df['Normalized Average RGB'] = (df['Average RGB'] - df['Average RGB'].min()) / (df['Average RGB'].max() - df['Average RGB'].min())

# Display the DataFrame with the normalized column
print(df)

# Create a dot plot for the normalized values
plt.figure(figsize=(12, 4))
plt.scatter(df['Pixel Index'], df['Normalized Average RGB'], color='gray', s=1)
plt.title('Pixel Index vs Normalized Average RGB (Dot Plot)')
plt.xlabel('Pixel Index')
plt.ylabel('Normalized Average RGB')
plt.tight_layout()
plt.show()

# Fourier Transform of the normalized values
y_values = df['Normalized Average RGB'].values

# Apply the Fourier Transform
fft_values = np.fft.fft(y_values)
frequencies = np.fft.fftfreq(len(y_values))

# Filter out high frequencies (optional)
threshold = 0.001  # Adjust this threshold as needed
fft_values_filtered = np.where(np.abs(frequencies) > threshold, 0, fft_values)

# Inverse Fourier Transform to get the smoothed pattern
smoothed_values = np.fft.ifft(fft_values_filtered)

# Plot the original noisy signal and the smoothed signal
plt.figure(figsize=(12, 4))
plt.plot(df['Pixel Index'], y_values, color='gray', alpha=0.5, label='Noisy Signal')
plt.plot(df['Pixel Index'], smoothed_values.real, color='blue', label='Extracted Pattern', linewidth=2)
plt.title('Noisy Signal with Extracted Pattern')
plt.xlabel('Pixel Index')
plt.ylabel('Normalized Average RGB')
plt.legend()
plt.tight_layout()
plt.show()

# Plot with vertical lines at every x=1000
plt.figure(figsize=(12, 4))
plt.plot(df['Pixel Index'], y_values, color='gray', alpha=0.5, label='Noisy Signal')
plt.plot(df['Pixel Index'], smoothed_values.real, color='blue', label='Extracted Pattern', linewidth=2)
for x in range(1000, len(df['Pixel Index']), 1000):
    plt.axvline(x=x, color='red', linestyle='--', linewidth=0.8)
plt.title('Noisy Signal with Vertical Lines')
plt.xlabel('Pixel Index')
plt.ylabel('Normalized Average RGB')
plt.legend()
plt.tight_layout()
plt.show()

# Plot with both x=100 and x=1000 vertical lines
plt.figure(figsize=(12, 4))
plt.plot(df['Pixel Index'], y_values, color='gray', alpha=0.5, label='Noisy Signal')
plt.plot(df['Pixel Index'], smoothed_values.real, color='blue', label='Extracted Pattern', linewidth=2)

# Add vertical red lines every x=100
for x in range(100, len(df['Pixel Index']), 100):
    plt.axvline(x=x, color='red', linestyle='--', linewidth=0.5)

# Add thicker lines and annotations every x=1000
for x in range(1000, len(df['Pixel Index']), 1000):
    plt.axvline(x=x, color='red', linestyle='-', linewidth=0.8)
    y_value = smoothed_values.real[x]
    plt.scatter(x, y_value, color='red')
    plt.text(x, y_value, f'{y_value:.2f}', color='red', ha='right', fontsize=8, rotation=45)

plt.title('Noisy Signal with Annotated Lines')
plt.xlabel('Pixel Index')
plt.ylabel('Normalized Average RGB')
plt.legend()
plt.tight_layout()
plt.show()