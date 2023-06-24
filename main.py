import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline

generator = torch.Generator(device="cuda").manual_seed(1024)

max_token = 49405
MODEL = "panopstor/EveryDream"
TOKEN_TARGET = 20

text_encoder = CLIPTextModel.from_pretrained(
    MODEL, subfolder="text_encoder"
).cuda()
vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae").cuda()
unet = UNet2DConditionModel.from_pretrained(
    MODEL, subfolder="unet"
).cuda()
tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")



img = Image.open("image.png").convert("RGB")
img = transforms.Resize((64, 64))(img)
img = transforms.ToTensor()(img).cuda()

with torch.no_grad():
    latents = vae.encode(img.unsqueeze(0)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    print(latents.shape)
latents.requires_grad_(True)
vae.requires_grad_(False)

sequence = torch.rand([TOKEN_TARGET - 2], device="cuda", generator=generator, requires_grad=True, dtype=torch.float)
#sequence *= 0.01
#sequence = sequence.float()
#sequence.requires_grad_(True)

criterion =  torch.nn.PoissonNLLLoss()

noise_scheduler = DDIMScheduler.from_pretrained(MODEL, subfolder="scheduler")  # type: ignore


def get_model_output(sequence):
    noise = torch.randn_like(latents)
    bsz = 1
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
    )
    timesteps = timesteps.long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    encoder_hidden_states = text_encoder(sequence.unsqueeze(0).long())[0]
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(
            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
        )

    return model_pred, target

optimizer = torch.optim.AdamW([sequence], lr=1e-1, weight_decay=1e-6, betas=(0.9,0.95))

token_init = torch.randint(0, max_token + 1, (TOKEN_TARGET - 2,), dtype=torch.float).cuda()

pipe = StableDiffusionPipeline.from_pretrained(MODEL).to("cuda")
for i in range(10000):
    #noise = torch.randn_like(sequence) * 1.0e-20
    #sequence = sequence + noise
    sequence = sequence.clamp(min=0, max=1)
    torch.nn.utils.clip_grad_norm_([sequence], 1.0)

    scaled = token_init * sequence
    scaled = torch.cat(
        [
            torch.tensor([max_token + 1]).cuda(),
            scaled.round(),
            torch.tensor([max_token + 2]).cuda(),
        ]
    )
    model_pred, target = get_model_output(scaled)
    loss = criterion(model_pred, target)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss.item())

    sequence_str_decoded = tokenizer.decode(scaled)

    if i % 100 == 0:
        print(sequence_str_decoded)
        images = pipe(
            prompt_embeds=text_encoder(scaled.unsqueeze(0).long())[0],
            guidance_scale=7.0,
            generator=generator,
        )  # type: ignore
        images[0][0].save(f"image_{i}.png")


sequence = sequence.detach().cpu().numpy().tolist()
sequence = sequence[1:-1]

sequence_str = tokenizer.decode(sequence)
with open("optimized_sequence.txt", "w") as f:
    f.write(sequence_str)
