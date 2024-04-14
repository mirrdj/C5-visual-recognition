import yaml
import os
import torch
from text_generation import create_captions
from image_generation import generate_image, get_pipe
from utils import move_file, read_json, read_txt

# Define what should be generated
classes = ["bus", "table", "cat", "house", "t-shirt", "dog", "pen", "cupcake"]
colors = ["blue", "red", "green", "yellow", "purple", "orange", "white", "black"]
prompts_per_pair = 50

# Define parameters for diffusion model
model ='2.1'
generator = torch.Generator("cuda").manual_seed(0)
pipe_args = {
    # "num_inference_steps": 30,
    # "generator": generator,
    # "guidance_scale": 0.0
}

with open('config.yml', 'r') as file:
    data = yaml.safe_load(file)

CAPTIONS_DIR = data['CAPTIONS_DIR']
UNGEN = data['UNGEN']
GEN = data['GEN']
IMG_DIR = data['IMG_DIR']
GENDB = data['GENERATED_DB']

gendb_full = os.path.join(IMG_DIR, GENDB)

dir_captions_ungenerated = os.path.join(CAPTIONS_DIR, UNGEN)
dir_captions_generated = os.path.join(CAPTIONS_DIR, GEN)

def extract_details(file_caption):
    object, color, _ = file_caption.split('_')
    
    return object, color

def text_step(dir_captions):
    for obj in classes:
        for color in colors:
            create_captions(dir_captions, obj, color, prompts_per_pair)


def image_step(dir_captions_ungenerated, dir_captions_generated):
    pipe = get_pipe(model)

    # Generate images based on the caption
    for file_caption in os.listdir(dir_captions_ungenerated):
        # Read caption from file
        captions = read_json(dir=dir_captions_ungenerated, file=file_caption)
        
        # Get color/object from the filename
        object, color = extract_details(file_caption)

        # Create directory for placing generated image
        img_dir = os.path.join(IMG_DIR, model, object, color)
        os.makedirs(img_dir, exist_ok=True)

        # Generate and save image + save in csv file
        for i, caption in enumerate(captions):
            print(f"Generate image with caption: {caption}")

            filename = f"{i}.png"
            filename = os.path.join(img_dir, filename)
            image = generate_image(pipe, pipe_args, caption)

            print(f"Save image at {filename}")
            image.save(filename)

            id = f"{object}_{color}_{i}"
            row = f"{id}: {caption} \n"

            # Using csv for now because it's easier to append data to it
            with open(gendb_full,'a') as file:
                file.write(row)

        move_file(file_caption, dir_captions_ungenerated, dir_captions_generated)    



def run_pipeline():
    # Generate captions
    text_step(dir_captions_ungenerated)

    # Generate images
    image_step(dir_captions_ungenerated, dir_captions_generated)
   
# text_step(dir_captions_ungenerated)
image_step(dir_captions_ungenerated, dir_captions_generated)
