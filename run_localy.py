from loaders.image_loader import create_img_loader, ImgLoaderTypes
from loaders.model_loader import ModelLoader

if __name__ == "__main__":
    img_path="/home/dubas/Pictures/tigercat.jpg"
    ocv_loader = create_img_loader(ImgLoaderTypes.OcvLoader)
    img_normalized = ocv_loader.load_and_normalize(img_path=img_path)

    model_loader = ModelLoader()
    model_loader.run_prediction(input_data=img_normalized)
