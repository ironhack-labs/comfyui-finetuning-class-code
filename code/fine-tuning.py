#pip install torch transformers diffusers accelerate
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import AutoTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, PNDMScheduler
from torch.optim import AdamW
from accelerate import Accelerator

# Configuración de la aceleración
accelerator = Accelerator()
device = accelerator.device

# Cargar el modelo y los componentes
model_name = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
pipe.to(device)

# Congelar las capas de VAE y UNet para ahorrar memoria, si es necesario
pipe.vae.requires_grad_(False)
pipe.unet.requires_grad_(False)

# Cargar el tokenizer y el modelo de texto
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_encoder = pipe.text_encoder
text_encoder.requires_grad_(True)  # Descongelar solo el text encoder para el ajuste fino

# Preparar los datos (deberás personalizar este bloque con tus propios datos)
def encode_texts(texts):
    inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
    return inputs.input_ids.to(device)

def load_images(images):
    return torch.stack([pipe.feature_extractor(images[i], return_tensors="pt").pixel_values.to(device) for i in range(len(images))])

# Configuración de optimización
optimizer = AdamW(text_encoder.parameters(), lr=5e-6)

# Bucle de entrenamiento
num_epochs = 3  # Ajusta esto según tus necesidades
for epoch in range(num_epochs):
    for step, (images, texts) in enumerate(dataloader):
        images = load_images(images)
        input_ids = encode_texts(texts)

        # Adelante
        latents = pipe.vae.encode(images).latent_dist.sample().detach()
        latents = latents * 0.18215  # Escalar latentes
        
        # Obtener el ruido
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # Predicción
        encoder_hidden_states = text_encoder(input_ids)[0]
        noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Calcular la pérdida
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # Backpropagation y optimización
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item()}")

    # Guardar el modelo ajustado
    pipe.save_pretrained(f"stable-diffusion-finetuned-epoch{epoch + 1}")

print("¡Ajuste fino completo!")
