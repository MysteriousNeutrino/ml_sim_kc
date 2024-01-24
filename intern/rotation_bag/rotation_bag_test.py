import rotation_bag_1
from skimage.data import stereo_motorcycle

original_image, _, _ = stereo_motorcycle()


def test_rotated_image():

    result = rotation_bag_1.rotated_image(original_image, 180)
    assert result.shape == original_image.shape, f"{result.shape} != {result.shape}"
