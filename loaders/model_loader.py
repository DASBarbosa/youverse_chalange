import onnxruntime as ort
import numpy as np
from pydantic import BaseModel
import time
from utils.time_utils import TimeMesure
from abc import ABC, abstractmethod
from enum import StrEnum


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


class ModelSettings(BaseModel):
    model_file_path: str
    labels_file_path: str


class ModelLoader(ABC):
    # Load class labels from .txt file
    def _load_labels(self, label_path: str) -> list[str]:
        with open(label_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        return labels

    def _from_score_to_prob(self, predictions: np.array) -> np.array:
        exp = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    # Enforcing an alternative constructor that expects a specific settings object is usefull to be able to implement
    # Other model loaders without changing any settings on the api
    @classmethod
    @abstractmethod
    def from_settings(cls, settings: ModelSettings):
        """Enforce creation from settings."""
        pass

    @abstractmethod
    def get_model_details(self) -> ModelDetails:
        pass

    @abstractmethod
    def run_prediction(self, input_data: np.array) -> ModelResponse:
        pass


class ModelLoaderTypes(StrEnum):
    OnnxLoader = "onnxloader"


class OnnxLoader(ModelLoader):
    def __init__(self, model_path: str, labels_path: str):
        # Load ONNX model
        self.ort_session = ort.InferenceSession(model_path)
        # Load labels
        self.labels = self._load_labels(label_path=labels_path)

    @classmethod
    def from_settings(cls, settings: ModelSettings):
        # Convert settings into whatever is needed for __init__
        return cls(
            model_path=settings.model_file_path, labels_path=settings.labels_file_path
        )

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
            ModelPredictions(
                label=self.labels[i], confidence=float(predictions_probs[i])
            )
            for i in range(len(self.labels))
        ]

        # sort by confidence descending
        all_predictions = sorted(
            all_predictions, key=lambda x: x.confidence, reverse=True
        )

        return ModelResponse(
            predictions=all_predictions, inference_time_ms=time_mes.elapsed_ms
        )


def create_model_loader(
    model_loader_type: ModelLoaderTypes,
    model_settings: ModelSettings | None,
    *args,
    **kwargs,
) -> ModelLoader:
    model_loader_map = {model_loader_type.OnnxLoader: OnnxLoader}
    try:
        if model_settings:
            # Prefer factory method if settings is given
            # it strongly types the inputs the factory receives
            return model_loader_map[model_loader_type].from_settings(
                settings=model_settings
            )
        else:
            # Fall back to standard constructor, and accepts any positional
            # or keyword arguments passed to the factory and injects them
            # to the standard construcor
            # might be overengeneering, we could just accept the strongly typed settings
            return model_loader_map[model_loader_type](*args, **kwargs)
    except KeyError:
        raise ValueError(f"Unsupported model loader: {model_loader_type}")
