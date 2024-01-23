import utils
from skimage.data import stereo_motorcycle

original_image, _, _ = stereo_motorcycle()


def test_rotated_image():

    result = utils.rotated_image(original_image, 180)
    assert original_image.shape == result.shape
