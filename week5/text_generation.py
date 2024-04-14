import os
import json
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from utils import contains_number

load_dotenv()
api_key = os.getenv("OPENAI_KEY")


def generate_captions(object, color, amount):
    prompt = f"Generate {amount} different descriptions for an image containing a {color} {object}. " \
    f" The description must be short up to 6 words, and the image showcases the {color} {object} in a common setting." \
    f"The description should contain the words {color} and  {object}"

    print(prompt)

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )

    captions = response.choices[0].message.content.split('\n')

    if contains_number(captions):
        captions = [cap.split('.')[1] for cap in captions]

    return captions

def save_json(dir, captions, object, color, amount):
    filename = f"{object}_{color}_{amount}.json"
    filename = os.path.join(dir, filename)

    with open(filename, 'w') as f :
        json.dump(captions, f)

    print(f"Saved captions for the {color} {object}!")


def create_captions(dir, object, color, amount):
    captions = generate_captions(object, color, amount)
    save_json(dir, captions, object, color, amount)


