import torch
from diffusers import StableDiffusionPipeline

# Cargar el modelo ajustado en la CPU
model_path = "stable-diffusion-finetuned-epoch3"  # Cambia este nombre si es necesario
pipe = StableDiffusionPipeline.from_pretrained(model_path).to("cpu")
# pipe = pipe.to("cpu") # Enviar el modelo a la CPU
# pipe = pipe.to("cuda")  # Enviar el modelo a la GPU

# Definir el prompt
prompt = "Una imagen de un perro Alaska Malamute en un bosque mágico"

# Generar la imagen
with torch.no_grad():
    result = pipe(prompt)
    image = result.images[0]

# Mostrar la imagen
image.show()
