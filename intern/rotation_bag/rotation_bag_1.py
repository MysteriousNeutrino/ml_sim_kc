"""
rotation bag
"""
import cv2
import numpy as np


def rotated_image(image: np.ndarray, angle: int = 45) -> np.ndarray:
    """Rotate image by angle degrees."""
    width, height, _ = image.shape
    transform = cv2.getRotationMatrix2D((height / 2, width / 2), angle, 1)
    # Apply the rotation to the image with a constant border
    result = cv2.warpAffine(image, transform, (height, width))

    return result

# # Загрузка изображения с помощью OpenCV
# image_path = "path/to/your/image.jpg"
# original_image, _, _ = stereo_motorcycle()
#
# # Вызов функции для вращения изображения
# rotated_image_result = rotated_image(original_image, angle=360)
#
# # Отображение оригинального и вращенного изображения
# cv2.imshow("Original Image", original_image)
# cv2.imshow("Rotated Image", rotated_image_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
