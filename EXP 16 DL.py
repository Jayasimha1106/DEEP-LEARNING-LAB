import numpy as np
import cv2
from matplotlib import pyplot as plt

# Image path
image_path = r"C:\Users\jayas\Downloads\pexels-christian-alvarez-116752650-12787369.jpg"

# Load image
img = cv2.imread(image_path)

# Check if image loaded successfully
if img is None:
    print("Error: Could not load image.")
    print("Please check the image path.")
else:
    # Convert BGR to RGB
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r, g, b])

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding
    ret, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Create kernel
    kernel = np.ones((2, 2), np.uint8)

    # Morphological closing
    closing = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Dilation
    sure_bg = cv2.dilate(
        closing,
        kernel,
        iterations=3
    )

    # Display all images
    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.imshow(rgb_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(232)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(233)
    plt.imshow(thresh, cmap="gray")
    plt.title("Otsu's Threshold")
    plt.axis("off")

    plt.subplot(234)
    plt.imshow(closing, cmap="gray")
    plt.title("Morphological Closing")
    plt.axis("off")

    plt.subplot(235)
    plt.imshow(sure_bg, cmap="gray")
    plt.title("Dilation")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Save dilation image
    cv2.imwrite(
        r"C:\Users\jayas\Downloads\dilation.png",
        sure_bg
    )

    print("Dilation image saved successfully!")