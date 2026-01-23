from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from loaders.image_loader import ImgLoaderTypes, create_img_loader
from loaders.model_loader import ModelLoader, ModelResponse

class InferenceAPI:
    def __init__(self, app: FastAPI):
        self.app = app
        self._register_routes()

    def _register_routes(self):
        self.app.get("/health")(self.health)
        self.app.post("/infer")(self.infer)

    def health(self):
        return {"status": "ok"}

    async def infer(
        self,
        number_of_predictions: int = Form(default=1),
        image: UploadFile = File(...)
    ) -> ModelResponse:
        image_bytes = await image.read()

        ocv_loader = create_img_loader(ImgLoaderTypes.OcvImageLoader)
        img_normalized = ocv_loader.load_img_bytes(image_bytes=image_bytes, height=224, width=224)

        model_loader = ModelLoader()
        model_response = model_loader.run_prediction(input_data=img_normalized)

        topk_predict = model_response.predictions[:number_of_predictions]
        for prediction in topk_predict:
            print(f"Label: {prediction.label}")
            print(f"Confidence: {prediction.confidence}")
        print(f"Inference time: {model_response.inference_time_ms} ms")
        model_response.predictions = topk_predict

        return model_response
