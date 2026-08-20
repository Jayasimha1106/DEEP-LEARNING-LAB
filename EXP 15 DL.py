import cv2
import numpy as np
from matplotlib import pyplot as plt

# --------------------------------
# Image Path
# --------------------------------
image_path = r"C:/Users/jayas\Downloads/pexels-christian-alvarez-116752650-12787369.jpg"

# --------------------------------
# Load Image
# --------------------------------
img = cv2.imread(image_path)

# Check if image was loaded successfully
if img is None:
    print("Error: Could not load image.")
    print("Please check the image path.")
else:

    print("Image loaded successfully!")

    # --------------------------------
    # Convert BGR to RGB
    # --------------------------------
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --------------------------------
    # Reshape image pixels
    # --------------------------------
    pixels = np.float32(
        rgb_img.reshape((-1, 3))
    )

    # --------------------------------
    # K-Means Criteria
    # --------------------------------
    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    # --------------------------------
    # Number of Clusters
    # --------------------------------
    K = 3

    # --------------------------------
    # Apply K-Means
    # --------------------------------
    _, labels, centers = cv2.kmeans(
        pixels,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    # --------------------------------
    # Convert Centers to uint8
    # --------------------------------
    centers = np.uint8(centers)

    # --------------------------------
    # Create Segmented Image
    # --------------------------------
    segmented_img = centers[
        labels.flatten()
    ].reshape(rgb_img.shape)

    # --------------------------------
    # Display Images
    # --------------------------------
    plt.figure(figsize=(12, 6))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Original Image")
    plt.axis("off")

    # Segmented Image
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_img)
    plt.title("Segmented Image (K-Means)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # --------------------------------
    # Print Results
    # --------------------------------
    print("\nK-Means Segmentation Completed")
    print("Number of Clusters:", K)
    print("Cluster Centers:")
    print(centers)