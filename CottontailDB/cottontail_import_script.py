import torch
import clip
import argparse
import os
from PIL import Image
from cottontaildb_client import CottontailDBClient, Type, Literal, column_def, float_vector


schema_name = 'images'
entity_name = 'image'

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(image_path):

    # prepare CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device)

    # connect to cottontail db
    with CottontailDBClient('localhost', 1865) as client, torch.no_grad():

        create_schema_and_entity(client)

        # detect if a file or folder is specified and act accordingly
        if os.path.isfile(image_path):
            abs_image_path, feature_vector = process_image(image_path, preprocess, model)
            insert_image(abs_image_path, feature_vector, client)

        elif os.path.isdir(image_path):
            num_images = len(os.listdir(image_path))
            for i, file in enumerate(os.listdir(image_path)):
                current_image_path = image_path + '/' + file
                abs_image_path, feature_vector = process_image(current_image_path, preprocess, model)
                insert_image(abs_image_path, feature_vector, client)
                if i % 10 == 0:
                    print(f'{i}/{num_images} done ({round(i / num_images, 2)}%)')


# extracts the path of the image from the root folder and calculates the images feature vector
def process_image(image_path, preprocess, model):
    abs_image_path = os.path.abspath(image_path)
    image = preprocess(Image.open(abs_image_path)).unsqueeze(0).to(device)
    [feature_vector] = model.encode_image(image).tolist()
    return abs_image_path, feature_vector


# puts an image with its feature vector into cottontail db
def insert_image(image_path, feature_vector, client):
    entry = {'image_path': Literal(stringData=image_path), 'feature_vector': float_vector(*feature_vector)}
    client.insert(schema_name, entity_name, entry)


def create_schema_and_entity(client):
    try:
        client.create_schema(schema_name)
        print("Schema has been created") 
    except:
        print("Schema already exist")

    try:
        columns = [
            column_def('image_path', Type.STRING, nullable=False),
            column_def('feature_vector', Type.FLOAT_VEC, length=512, nullable=False)
        ]

        client.create_entity(schema_name, entity_name, columns)
        print("Entity has been created")
    except:
        print("Entity already exists")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script applies the CLIP model to images and writes them to a Cottontail database'
        )
    parser.add_argument('-i', '--image', type=str, help='Image or folder containing images')

    args = parser.parse_args()
    main(args.image)
