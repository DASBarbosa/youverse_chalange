import numpy as np
from abc import ABC, abstractmethod
from enum import StrEnum
import cv2

# Decided to create a factory because it makes it easier to add new types of loaders
# for now only 2 were added, one using open cv to test the model localy,
# and another for usin direct bytes so it can be used on the fastapi setup
# This might be a bit overengineering for this exercice


class ImageLoader(ABC):

    @abstractmethod
    def load_img(self, img_path: str, height: int, width: int) -> np.array:
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

    def load_and_normalize(
        self,
        img_path: str,
        height: int = 224,
        width: int = 224,
        channels: int = 3,
        batch: int = 1,
    ) -> np.array:
        img = self.load_img(img_path=img_path, height=height, width=width)
        return self.normalize_image(
            input_data=img, height=height, width=width, channels=channels, batch=batch
        )


class ImgLoaderTypes(StrEnum):
    OcvLoader = "ocvloader"


class OcvImgLoader(ImageLoader):

    def load_img(self, img_path: str, height: int, width: int) -> np.array:
        image = cv2.imread(img_path)
        # OpenCv loads image as BGR and not RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        # Convert to float32
        image = image.astype(np.float32)
        # changes HWC to CHW which is what the onnx interface expects
        image = np.transpose(image, (2, 0, 1))

        return image


# Factory function for an image loder
def create_img_loader(loader_type: ImgLoaderTypes) -> ImageLoader:
    loader_map = {ImgLoaderTypes.OcvLoader: OcvImgLoader}
    try:
        return loader_map[loader_type]()
    except KeyError:
        raise ValueError(f"Unsupported image loader: {loader_type}")

if __name__ == "__main__" :
    ocv_loader = create_img_loader(ImgLoaderTypes.OcvLoader)
    ocv_loader.load_and_normalize(img_path="/home/dubas/Pictures/tigercat.jpg")
