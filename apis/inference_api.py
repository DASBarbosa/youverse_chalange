from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from loaders.image_loader import ImgLoaderTypes, create_img_loader
from loaders.model_loader import ModelLoaderTypes, ModelResponse, create_model_loader


class InferenceAPI:
    def __init__(
        self,
        app: FastAPI,
        img_loader_type: ImgLoaderTypes,
        model_loader_type: ModelLoaderTypes,
    ):
        self.app = app
        self._img_loader_type = img_loader_type
        self._model_loader_type = model_loader_type
        self.model_ready: bool = False
        self.loading_error: str | None = None
        self._register_routes()

    def _register_routes(self):
        self.app.get("/health")(self.health)
        self.app.post("/infer")(self.infer)
        self.app.add_event_handler("startup", self.on_startup)

    def on_startup(self):
        # On startup will load the model and img loaders at startup tipe
        # if there is anything wrong with any of them, will set model_ready to false
        try:
            self.image_loader = create_img_loader(loader_type=self._img_loader_type)
            self.model_loader = create_model_loader(
                model_loader_type=self._model_loader_type
            )
            self.model_ready = True
        except Exception as e:
            self.model_ready = False
            self.loading_error = str(e)

    def health(self):
        return {
            "status": "ok",
            "model_loaded": self.model_ready,
            "loading_error": self.loading_error
        }

    async def infer(
        self,
        number_of_predictions: int = Form(default=1),
        image: UploadFile = File(...),
    ) -> ModelResponse:
        image_bytes = await image.read()
        img_normalized = self.image_loader.load_img_bytes(
            image_bytes=image_bytes, height=224, width=224
        )
        model_response = self.model_loader.run_prediction(input_data=img_normalized)

        topk_predict = model_response.predictions[:number_of_predictions]
        for prediction in topk_predict:
            print(f"Label: {prediction.label}")
            print(f"Confidence: {prediction.confidence}")
        print(f"Inference time: {model_response.inference_time_ms} ms")
        model_response.predictions = topk_predict

        return model_response
