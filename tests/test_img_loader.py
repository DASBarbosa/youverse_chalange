from loaders.image_loader import create_img_loader,ImgLoaderTypes, OcvImgLoader
import numpy as np
import pytest

@pytest.fixture
def mock_image_np():
    return np.array(
        [
            [[0, 0, 0], [255, 255, 255]],
            [[128, 128, 128], [64, 64, 64]],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def expected_normalized_np():
    """
    Expected normalized image for a 2x2 RGB image after:
    1) Converting to float [0,1]
    2) Standard mean/std normalization
    3) Reshaping to ONNX format (NCHW)

    using mock_image_np as Original image

    Steps:
    ---------
    1. Convert to float in range [0,1]:
       image_float = image_uint8 / 255.0

    2. Apply per-channel normalization:
       normalized = (image_float - mean) / std

       Where:
       mean = [0.485, 0.456, 0.406]  # RGB
       std  = [0.229, 0.224, 0.225]

       Example for pixel [255,255,255]:
       R: (1.0 - 0.485)/0.229 ≈ 2.2489
       G: (1.0 - 0.456)/0.224 ≈ 2.4286
       B: (1.0 - 0.406)/0.225 ≈ 2.6400

       Example for pixel [0,0,0]:
       R: (0.0 - 0.485)/0.229 ≈ -2.1179
       G: (0.0 - 0.456)/0.224 ≈ -2.0357
       B: (0.0 - 0.406)/0.225 ≈ -1.8044

       and so on ...

    3. Doing manual calculations this is how the original normalized HWC image looks:
         [[-2.1179, -2.0357, -1.8044], [2.2489, 2.4286, 2.6400]],
         [[0.0741, 0.2052, 0.4265], [-1.0219, -0.9153, -0.6890]]
       ]

    4. Converting it to CHW (Channels first) looks like this:
    """
    return np.array(
        [
            [[-2.117904,  2.2489083],
             [ 0.07406463, -1.0219197]],

            [[-2.0357141,  2.4285715],
             [ 0.2051822, -0.91526604]],

            [[-1.8044444,  2.64],
             [ 0.42649257, -0.68897593]]
        ],
        dtype=np.float32
    )

def test_factory_returns_expected_class():
    #Note if we had more classes in the factory this could be improved with pytest parametrize
    ocv_loader = create_img_loader(loader_type=ImgLoaderTypes.OcvImageLoader)
    assert type(ocv_loader) == OcvImgLoader

def test_image_is_normalized_using_mean_std(mock_image_np, expected_normalized_np):
    ocv_loader = create_img_loader(loader_type=ImgLoaderTypes.OcvImageLoader)
    # changes HWC to CHW
    transposed_img = np.transpose(mock_image_np, (2, 0, 1))
    normalized_image = ocv_loader.normalize_image(input_data=transposed_img, height=2, width=2)
    # reshape expected np array before comparing to match the shape of the return from the normalize method
    reshaped_expected_np = expected_normalized_np.reshape(1, 3, 2, 2)

    # atol: absolute tolerance
    # rtol: relative tolerance
    # these parameters above help the test pass if every element is very close, accounting for floating point rounding
    # thrse is very ussefull when working with small numbers because erros can be amplified
    np.testing.assert_allclose(normalized_image, reshaped_expected_np, rtol=1e-5, atol=1e-5)