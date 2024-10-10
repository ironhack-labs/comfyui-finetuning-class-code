# .safetensor to .ckpt :) 
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file as load_safetensors
import torch

# Ruta del modelo safetensors que deseas convertir
# safetensors_path = "origin path"
# ckpt_path = "destiny path"
safetensors_path = "origin path"
ckpt_path = "destiny path"

# Load safetensor
weights = load_safetensors(safetensors_path)

# Save in ckpt format
torch.save(weights, ckpt_path)

print(f"Conversión completada. El modelo se ha guardado en {ckpt_path}")
