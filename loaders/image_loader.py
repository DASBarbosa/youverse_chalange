import numpy as np
from abc import ABC, abstractmethod
from enum import StrEnum
import cv2

# Decided to create a factory because it makes it easier to add new types of loaders
# for now only 1 were added using open cv to test the model localy, and for loading images from bytes
# For this particular exercice this might be overeengenering, but with a factory, we can easily
# change the library or meodology we use for loading an image without having to change anything
# in the api itself.


class ImageLoader(ABC):

    @abstractmethod
    def load_local_img(
        self, img_path: str, height: int, width: int, channels: int, batch: int
    ) -> np.array:
        pass

    @abstractmethod
    def load_img_bytes(
        self, image_bytes: bytes, height: int, width: int, channels: int, batch: int
    ) -> np.array:
        pass

    def normalize_image(
        self,
        input_data: np.array,
        height: int = 224,
        width: int = 224,
        channels: int = 3,
        batch: int = 1,
    ) -> np.array:
        img = input_data.astype(np.float32) / 255.0

        # acording to this source, this seems to be the normalization expression that the resnet50 model expects:
        # https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/resnet50_modelzoo_onnxruntime_inference.ipynb

        # Reshape to match the shape format of the image, aka C,H,W
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        # Reshape image data to add batch at the begining and set the H and W to 224
        return img.reshape(batch, channels, height, width)


class ImgLoaderTypes(StrEnum):
    OcvImageLoader = "ocvImageloader"


class OcvImgLoader(ImageLoader):

    def load_img_bytes(
        self,
        image_bytes: bytes,
        height: int,
        width: int,
        channels: int = 3,
        batch: int = 1,
    ) -> np.array:
        # Decode bytes → BGR image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image bytes")

        # OpenCv loads image as BGR and not RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        # changes HWC to CHW which is what the onnx interface expects
        image = np.transpose(image, (2, 0, 1))

        return self.normalize_image(
            input_data=image, height=height, width=width, channels=channels, batch=batch
        )

    def load_local_img(
        self, img_path: str, height: int, width: int, channels: int = 3, batch: int = 1
    ) -> np.array:
        image = cv2.imread(img_path)
        # OpenCv loads image as BGR and not RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        # changes HWC to CHW which is what the onnx interface expects
        image = np.transpose(image, (2, 0, 1))

        return self.normalize_image(
            input_data=image, height=height, width=width, channels=channels, batch=batch
        )


# Factory function for an image loder
def create_img_loader(loader_type: ImgLoaderTypes) -> ImageLoader:
    loader_map = {ImgLoaderTypes.OcvImageLoader: OcvImgLoader}
    try:
        return loader_map[loader_type]()
    except KeyError:
        raise ValueError(f"Unsupported image loader: {loader_type}")
