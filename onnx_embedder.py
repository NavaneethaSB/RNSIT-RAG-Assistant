
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class ONNXEmbedder:
    def __init__(self, model_path="onnx_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            f"{model_path}/model.onnx",
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )


    def get_embedding_dimension(self):
        return self.session.get_outputs()[0].shape[-1]
    
    
    def encode(
        self,
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True
    ):
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="np"
        )

        feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }

        # Include token_type_ids only when required by the ONNX model.
        required_inputs = {
            item.name for item in self.session.get_inputs()
        }

        if "token_type_ids" in required_inputs:
            feed["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

        outputs = self.session.run(None, feed)
        token_embeddings = outputs[0]

        mask = inputs["attention_mask"][..., None].astype(np.float32)

        embeddings = np.sum(token_embeddings * mask, axis=1)
        embeddings /= np.clip(
            mask.sum(axis=1), a_min=1e-9, a_max=None
        )

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings /= np.clip(norms, a_min=1e-12, a_max=None)

        if convert_to_numpy:
            return embeddings.astype(np.float32)

        return embeddings
