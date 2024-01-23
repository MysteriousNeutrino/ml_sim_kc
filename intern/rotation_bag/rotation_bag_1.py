"""
rotation bag
"""
import cv2
import numpy as np
from skimage.data import stereo_motorcycle, vortex


import cv2
import numpy as np

def rotated_image(image: np.ndarray, angle: int = 45) -> np.ndarray:
    """Rotate image by angle degrees."""
    width, height, channels = image.shape
    print(width, height, image.shape[:2])
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    border_value = (255, 255, 255)  # White color in BGR format

    # Apply the rotation to the image with a constant border
    result = cv2.warpAffine(original_image, transform, (width, height), borderValue=border_value,
                                   borderMode=cv2.BORDER_CONSTANT)


    return result

# Загрузка изображения с помощью OpenCV
image_path = "path/to/your/image.jpg"
original_image, _, _ = stereo_motorcycle()

# Вызов функции для вращения изображения
rotated_image_result = rotated_image(original_image, angle=360)

# Отображение оригинального и вращенного изображения
cv2.imshow("Original Image", original_image)
cv2.imshow("Rotated Image", rotated_image_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
