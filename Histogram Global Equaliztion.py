import cv2

input_path = "C:\\Users\\usaal\\OneDrive\\Desktop\\images\\test-img6.pgm"
output_path = "C:\\Users\\usaal\\OneDrive\\Desktop\\test-img6-global-eq.pgm"

img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not load the image.")
    exit()

# Apply global histogram equalization
equalized = cv2.equalizeHist(img)

cv2.imwrite(output_path, equalized)

print(f"Equalized image saved to: {output_path}")