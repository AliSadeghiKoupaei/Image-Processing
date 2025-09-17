import cv2
import numpy as np

def detect_shadow_mask(image_gray):
    blur = cv2.GaussianBlur(image_gray, (21, 21), 0)
    shadow_mask = image_gray < (blur - 10)
    return shadow_mask.astype(np.uint8)

def enhance_crosswalk_local(image_path, save_path=None, use_highpass=False):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shadow_mask = detect_shadow_mask(gray)

    # Illumination correction locally
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    illum_corrected = gray.copy()
    illum_corrected = cv2.divide(gray, blur, scale=255)
    illum_corrected[shadow_mask == 0] = gray[shadow_mask == 0]

    # CLAHE locally
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = illum_corrected.copy()
    clahe_result = clahe.apply(illum_corrected)
    enhanced[shadow_mask == 1] = clahe_result[shadow_mask == 1]

    if use_highpass:
        # High-pass filter globally
        f = np.fft.fft2(enhanced)
        fshift = np.fft.fftshift(f)
        rows, cols = enhanced.shape
        crow, ccol = rows // 2 , cols // 2
        mask = np.ones((rows, cols), np.uint8)
        r = 20
        mask[crow-r:crow+r, ccol-r:ccol+r] = 0
        fshift = fshift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        final_output = np.abs(img_back)
        final_output = cv2.normalize(final_output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        final_output = enhanced

    # Save result
    if save_path:
        cv2.imwrite(save_path, final_output)

    return final_output

# Automatically call the function when script runs
enhance_crosswalk_local(
    image_path=r'C:/Users/usaal/OneDrive/Desktop/satellite_image.png',
    save_path=r'C:/Users/usaal/OneDrive/Desktop/outputt1.png',
    use_highpass=True
)
