from loaders.image_loader import create_img_loader, ImgLoaderTypes
from loaders.model_loader import ModelLoader
from utils.stdin_loader import read_stdin

if __name__ == "__main__":
    img_path = read_stdin()
    if len(img_path) == 0:
        raise ValueError("Image path cannot be empty")

    ocv_loader = create_img_loader(ImgLoaderTypes.OcvImageLoader)
    img_normalized = ocv_loader.load_local_img(img_path=img_path, height=224, width=224)

    model_loader = ModelLoader()
    model_response = model_loader.run_prediction(input_data=img_normalized)

    # top 5 example
    top5_predict = model_response.predictions[:5]
    for prediction in top5_predict:
        print(f"Label: {prediction.label}")
        print(f"Confidence: {prediction.confidence}")
    print(f"Inference time: {model_response.inference_time_ms} ms")
