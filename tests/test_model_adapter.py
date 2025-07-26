import torch
from toolkit.modules.model_adapter import ModelAdapter

def test_model_adapter():
    adapter = ModelAdapter(
        base_model_name="NousResearch/Llama-2-7b-chat-hf",
        num_labels=2,
        lora_rank=4
    )
    texts = ["Positive example.", "Negative example."]
    logits = adapter(texts)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 2)
    