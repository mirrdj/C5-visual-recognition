from diffusers import AutoPipelineForText2Image
from PIL import Image
import torch

model_id='stabilityai/stable-diffusion-2-1'
prompt = ['a red bird in a tree']
output_file = 'red_birds.png'

def get_model_id(model):
    if model == 'XL':
        return 'stabilityai/stable-diffusion-xl-base-1.0'
    elif model == '2.1':
        return 'stabilityai/stable-diffusion-2-1'
    elif model == '2.1_turbo':
        return 'stabilityai/sd-turbo'
    elif model == 'XL_turbo':
        return 'stabilityai/sdxl-turbo'

def get_pipe(model, variant=None):
    model_id = get_model_id(model)
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.bfloat16, variant=variant).to('cuda')

    return pipe

def generate_image(pipe, pipe_args, prompt):
    negative_prompt = "multiple, bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms," \
                      "poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, " \
                      "fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, " \
                      "worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, huge eyes, 2girl, " \
                      "amputation, disconnected limbs, cartoon, cg, 3d, unreal, animate, anime"

    prompt += " realistic "

    image = pipe(prompt, negative_prompt=negative_prompt, **pipe_args).images[0]
    return image

def save_image():
    pass
