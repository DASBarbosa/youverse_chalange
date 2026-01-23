from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from loaders.image_loader import ImgLoaderTypes, create_img_loader
from loaders.model_loader import (
    ModelLoaderTypes,
    ModelResponse,
    ModelSettings,
    create_model_loader,
)
from pydantic_settings import BaseSettings


# Base settings are pretty handy, not only they will load these variables from env variables
# if available, but they also make test easier, since you can just inject a settings object
class InferenceApiSettings(BaseSettings):
    IMAGE_LOADER_TYPE: ImgLoaderTypes | None
    MODEL_LOADER_TYPE: ModelLoaderTypes | None
    MODEL_PATH: str
    LABELS_PATH: str
    TOP_K: int
    NUM_THREADS: int


class InferenceAPI:
    def __init__(
        self,
        app: FastAPI,
        api_settings: InferenceApiSettings,
    ):
        self.app = app
        self._settings = api_settings
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
        if (
            self._settings.IMAGE_LOADER_TYPE == None
            or self._settings.MODEL_LOADER_TYPE == None
        ):
            self.model_ready = False
            self.loading_error = "no loader for model or image was provided"
            return
        try:
            self.image_loader = create_img_loader(
                loader_type=self._settings.IMAGE_LOADER_TYPE
            )
            self.model_loader = create_model_loader(
                model_loader_type=self._settings.MODEL_LOADER_TYPE,
                model_settings=ModelSettings(
                    model_file_path=self._settings.MODEL_PATH,
                    labels_file_path=self._settings.LABELS_PATH,
                ),
            )
            self.model_ready = True
        except Exception as e:
            self.model_ready = False
            self.loading_error = str(e)

    def health(self):
        return {
            "status": "ok",
            "model_loaded": self.model_ready,
            "loading_error": self.loading_error,
        }

    async def infer(
        self,
        image: UploadFile = File(...),
    ) -> ModelResponse:
        image_bytes = await image.read()
        img_normalized = self.image_loader.load_img_bytes(
            image_bytes=image_bytes, height=224, width=224
        )
        model_response = self.model_loader.run_prediction(input_data=img_normalized)

        topk_predict = model_response.predictions[: self._settings.TOP_K]
        for prediction in topk_predict:
            print(f"Label: {prediction.label}")
            print(f"Confidence: {prediction.confidence}")
        print(f"Inference time: {model_response.inference_time_ms} ms")
        model_response.predictions = topk_predict

        return model_response
