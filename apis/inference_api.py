from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse


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
        number_of_predictions: int = Form(...),
        image: UploadFile = File(...)
    ):
        image_bytes = await image.read()

        predictions = [
            f"prediction_{i}"
            for i in range(number_of_predictions)
        ]

        return JSONResponse(
            content={
                "filename": image.filename,
                "num_predictions_requested": number_of_predictions,
                "predictions": predictions,
            }
        )