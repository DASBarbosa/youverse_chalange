import onnxruntime as ort
import numpy as np
from pydantic import BaseModel
import time
from utils.time_utils import TimeMesure


class ModelDetails(BaseModel):
    input_name: str
    input_shape: list[str | int | float]
    output_name: str


class ModelPredictions(BaseModel):
    label: str
    confidence: float


class ModelResponse(BaseModel):
    predictions: list[ModelPredictions]
    inference_time_ms: float


class ModelLoader:
    def __init__(self):
        # Load ONNX model
        self.ort_session = ort.InferenceSession("resnet50.onnx")
        # Load labels
        self.labels = self._load_labels(label_path="imagenet_classes.txt")

    # Load class labels from .txt file
    def _load_labels(self, label_path: str) -> list[str]:
        with open(label_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        return labels

    def _from_score_to_prob(self,predictions:np.array) -> np.array:
        exp = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)


    def get_model_details(self) -> ModelDetails:
        return ModelDetails(
            input_name=self.ort_session.get_inputs()[0].name,
            input_shape=self.ort_session.get_inputs()[0].shape,
            output_name=self.ort_session.get_outputs()[0].name,
        )


    def run_prediction(self, input_data: np.array) -> ModelResponse:
        model_details = self.get_model_details()

        with TimeMesure() as time_mes:
            # Run inference
            outputs = self.ort_session.run(
                [model_details.output_name], {model_details.input_name: input_data}
            )

        # Get predictions
        # predictions: shape (1, 1000) acording to this model
        predictions_probs = self._from_score_to_prob(predictions=outputs[0])
        # flatten batch 0 (since batch_size=1)
        predictions_probs = predictions_probs[0]

        # all predictions as type of ModelPredictions
        all_predictions = [
            ModelPredictions(label=self.labels[i], confidence=float(predictions_probs[i]))
            for i in range(len(self.labels))
        ]

        #sort by confidence descending
        all_predictions = sorted(all_predictions, key=lambda x: x.confidence, reverse=True)


        return ModelResponse(
            predictions = all_predictions,
            inference_time_ms = time_mes.elapsed_ms
        )

