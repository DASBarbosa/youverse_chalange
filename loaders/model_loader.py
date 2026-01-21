import onnxruntime as ort
import numpy as np
from pydantic import BaseModel


class ModelDetails(BaseModel):
    input_name: str
    input_shape: list[str | int | float]
    output_name: str


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

    def get_model_details(self) -> ModelDetails:
        return ModelDetails(
            input_name=self.ort_session.get_inputs()[0].name,
            input_shape=self.ort_session.get_inputs()[0].shape,
            output_name=self.ort_session.get_outputs()[0].name,
        )

    def run_prediction(self, input_data: np.array):
        model_details = self.get_model_details()

        # Run inference
        outputs = self.ort_session.run(
            [model_details.output_name], {model_details.input_name: input_data}
        )

        # Get predictions
        predictions = outputs[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_class_label = self.labels[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]

        print(f"Predicted class: {predicted_class_label}")
        print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    batch_size = 1
    random_input = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)

    model_loader = ModelLoader()
    model_loader.run_prediction(input_data=random_input)
