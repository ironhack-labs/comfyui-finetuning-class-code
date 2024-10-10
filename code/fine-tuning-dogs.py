import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer
from torch.optim import AdamW
from accelerate import Accelerator
from PIL import Image
import os

# Inicializa el entorno de aceleración
accelerator = Accelerator()
device = accelerator.device

# Cargar el modelo Stable Diffusion en float32
model_name = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_name).to(device)

# Congelar VAE y UNet para ahorrar memoria
pipe.vae.requires_grad_(False)
pipe.unet.requires_grad_(False)

# Tokenizer y encoder de texto
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
text_encoder.requires_grad_(True)  # Solo ajustar el text encoder

# Función para cargar las imágenes y los textos
def load_images_and_texts(image_folder):
    images = []
    texts = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            img = Image.open(os.path.join(image_folder, filename)).convert("RGB")
            img = pipe.feature_extractor(images=img, return_tensors="pt").pixel_values.to(device)
            images.append(img)
            
            text_filename = filename.replace(".jpg", ".txt")
            with open(os.path.join(image_folder, text_filename), "r") as f:
                text = f.read().strip()
                texts.append(text)
    
    # Usamos torch.stack para combinar las imágenes en lugar de torch.cat y evitar problemas de tamaño.
    return torch.stack(images), texts

# Cargar las imágenes y textos
image_folder = "../assets/fine-tuning-imgs_resized"
images, texts = load_images_and_texts(image_folder)

# Configuración del optimizador
optimizer = AdamW(text_encoder.parameters(), lr=5e-6)

# Bucle de entrenamiento
num_epochs = 3  # Ajusta según necesites
for epoch in range(num_epochs):
    for step in range(len(texts)):
        image = images[step].to(device)
        text = texts[step]

        # Codificación del texto
        input_ids = tokenizer(text, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(device)

        # Latentes y ruido
        latents = pipe.vae.encode(image).latent_dist.sample().detach()
        latents = latents * 0.18215
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # Predicción
        encoder_hidden_states = text_encoder(input_ids)[0]
        noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Cálculo de la pérdida
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # Backpropagation y optimización
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item()}")

    # Guarda el modelo después de cada época
    pipe.save_pretrained(f"stable-diffusion-finetuned-epoch{epoch + 1}")

print("¡Fine-tuning completado!")
