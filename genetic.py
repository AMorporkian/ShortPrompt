# A genetic algorithm for finding the shortest prompt that satisfies the minimum loss distance between the prompt and the target.

import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline
from torchmetrics import StructuralSimilarityIndexMeasure

with open("words.txt") as f:
    words = f.read().splitlines()
torch.set_default_device("cuda")
generator = torch.Generator(device="cuda").manual_seed(1024)

max_token = 49405
MODEL = "panopstor/EveryDream"
TOKEN_MAX = 40

text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").cuda()

vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae").cuda()

unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet").cuda()

tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")

img = Image.open("image.png").convert("RGB")
img = transforms.ToTensor()(img).cuda()

with torch.no_grad():
    target_latent = vae.encode(img.unsqueeze(0)).latent_dist.sample()
    target_latent = target_latent * vae.config.scaling_factor

vae.requires_grad_(False)
# criterion =  torch.nn.MSELoss()
criterion = StructuralSimilarityIndexMeasure(reduction="none")
noise_scheduler = DDIMScheduler.from_pretrained(MODEL, subfolder="scheduler")  # type: ignore


def get_model_output(sequence):
    bsz = sequence.shape[0]
    timesteps = torch.Tensor([1000 for _ in range(bsz)]).to(
        dtype=torch.long, device="cuda"
    )
    # noisy_latents = noise_scheduler.add_noise(target_latent, noise, timesteps)
    s = sequence.long()
    encoder_hidden_states = text_encoder(s)[0]
    t = target_latent.expand(
        bsz, -1, -1, -1
    )  # expand to match the number of dimensions in model_pred
    model_pred = unet(t, timesteps, encoder_hidden_states).sample
    return model_pred


import random


def choose_word():
    return [random.randint(0, max_token)]


def create_starting_sequence():
    sequence = choose_word()
    return sequence


def mutate(sequence):
    import copy

    # Either change a random token, add a random token, remove a random token, or swap two random tokens.
    mutation_type = random.randint(0, 3)
    sequence = copy.copy(sequence)
    if mutation_type == 0:
        sequence = change_token(sequence)
    elif mutation_type == 1:
        sequence = add_token(sequence)
    elif mutation_type == 2:
        sequence = remove_token(sequence)
    elif mutation_type == 3:
        sequence = swap_token(sequence)
    return sequence


def change_token(sequence):
    # Change a random token to a random token.
    idx = random.randint(0, len(sequence) - 1)
    sequence = sequence[:idx] + choose_word() + sequence[idx + 1 :]
    return sequence


def add_token(sequence):
    # Add a random token to a random position.
    idx = random.randint(0, len(sequence) - 1)
    sequence = sequence[:idx] + choose_word() + sequence[idx:]
    return sequence


def remove_token(sequence):
    if len(sequence) == 1:
        return change_token(sequence)
    # Remove a random token.
    idx = random.randint(0, len(sequence) - 1)
    sequence.pop(idx)
    return sequence


def swap_token(sequence):
    # Swap two random tokens.
    idx1 = random.randint(0, len(sequence) - 1)
    idx2 = random.randint(0, len(sequence) - 1)
    sequence[idx1], sequence[idx2] = sequence[idx2], sequence[idx1]
    return sequence


def crossover(sequence1, sequence2):
    # Crossover two sequences by randomly swapping tokens.
    idx = random.randint(0, len(sequence1) - 2)
    sequence1[idx], sequence2[idx] = sequence2[idx], sequence1[idx]
    return sequence1, sequence2


def get_sequence_loss(sequence):
    sequence = torch.tensor([sequence]).cuda()
    with torch.no_grad():
        model_pred = get_model_output(sequence)
        loss = criterion(model_pred, target_latent).mean()
    return loss


fitness_cache = {}


def get_fitness(sequence):
    if hash(str(sequence)) in fitness_cache:
        return fitness_cache[hash(str(sequence))]
    else:
        loss = get_sequence_loss(sequence)
        fitness = 1.0 / loss
        fitness_cache[hash(str(sequence))] = fitness
        return fitness


def get_population_fitness(population, BATCH_SIZE=64):
    fitnesses = []
    for i in range(0, len(population), BATCH_SIZE):
        batch = population[i : i + BATCH_SIZE]
        tokenizer.pad_token = tokenizer.eos_token
        # Pad the sequences to the same length
        max_len = max(len(x) for x in batch)
        for i in range(len(batch)):
            batch[i].extend([tokenizer.pad_token_id] * (max_len - len(batch[i])))

        batch_tensor = torch.tensor(batch).cuda()
        with torch.no_grad():
            model_pred = get_model_output(batch_tensor)
            losses = criterion(
                model_pred, target_latent.expand(batch_tensor.shape[0], -1, -1, -1)
            )
            fitnesses.extend([1.0 / loss for loss in losses])

    # Cache the fitnesses for individual sequences
    for i in range(len(population)):
        fitness_cache[hash(str(population[i]))] = fitnesses[i]
    return fitnesses


def get_best_sequence(population):
    fitnesses = get_population_fitness(population)
    best_idx = fitnesses.index(max(fitnesses))
    return [x for x in population[best_idx] if x <= 49405]


def get_worst_sequence(population):
    fitnesses = get_population_fitness(population)
    worst_idx = fitnesses.index(min(fitnesses))
    return population[worst_idx]


def get_average_fitness(population):
    fitnesses = get_population_fitness(population)
    return sum(fitnesses) / len(fitnesses)


def get_population_variance(population):
    fitnesses = get_population_fitness(population)
    mean = sum(fitnesses) / len(fitnesses)
    variance = sum((x - mean) ** 2 for x in fitnesses) / len(fitnesses)
    return variance


def run_generation(population, needed=1):
    new_population = []
    while len(new_population) < needed:
        for i in range(len(population)):
            new_sequence = mutate(population[i])
            new_population.append(new_sequence)
    return new_population


def generate_best_sample(population, i=0):
    best_sequence = get_best_sequence(population)
    generate_sample(best_sequence, i)


def generate_sample(sequence, i=0):
    prompt = tokenizer.decode(sequence)
    images = pipe(
        prompt,
        guidance_scale=7.0,
        generator=generator,
    )  # type: ignore
    import ftfy

    prompt = ftfy.fix_text(prompt)
    images[0][0].save(f"output/{i}_{prompt}.png")
    print(prompt)


def create_population(n):
    population = []
    for i in range(n):
        population.append(create_starting_sequence())
    return population


def create_population_from_prompt(prompt, n):
    population = []
    for i in range(n):
        population.append(tokenizer.encode(prompt)[1:-1])
    return population


def get_population_stats(population):
    best_sequence = get_best_sequence(population)
    worst_sequence = get_worst_sequence(population)
    average_fitness = get_average_fitness(population)
    variance = get_population_variance(population)
    return best_sequence, worst_sequence, average_fitness, variance


def selection(population):
    # Select the top 50% of the population
    population.sort(key=lambda x: get_fitness(x), reverse=True)
    return population[: len(population) // 10]


if __name__ == "__main__":
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, safety_checker=None).to(
        "cuda"
    )
    population = create_population_from_prompt("an emoji bicep", 512)
    for i in range(1000):
        # Randomly cross over two sequences
        # idx1 = random.randint(0, len(population)-1)
        # idx2 = random.randint(0, len(population)-1)
        # population[idx1], population[idx2] = crossover(population[idx1], population[idx2])

        # Sort the population by fitness
        population.sort(key=lambda x: get_fitness(x), reverse=True)

        # Print stats

        best_sequence, worst_sequence, average_fitness, variance = get_population_stats(
            population
        )
        print(f"Generation {i}")
        print(f"Best sequence: {best_sequence}")
        print(f"Worst sequence: {worst_sequence}")
        print(f"Average fitness: {average_fitness}")
        print("Best sequence loss:", get_sequence_loss(best_sequence))
        print("Worst sequence loss:", get_sequence_loss(worst_sequence))
        print("Average loss:", get_average_fitness(population))
        print(f"Variance: {variance}")
        if i % 1 == 0:
            generate_best_sample(population, i)

        # Hold on to the best sequence
        parents = selection(population)
        # Generate new population
        children = run_generation(parents, len(population) - len(parents))
        population = parents + children
