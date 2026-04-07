# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("fares-boutriga/Damork", dtype="auto")