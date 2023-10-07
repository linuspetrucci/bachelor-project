import socket
import torch
import clip
import argparse
import os
import io
from PIL import Image
from cottontaildb_client import CottontailDBClient, Type, Literal, column_def, float_vector


schema_name = 'images'
entity_name = 'image'

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(port):

    model, preprocess = clip.load("ViT-B/32", device=device)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, CottontailDBClient('localhost', 1865) as client, torch.no_grad():
        s.bind(("", port))
        s.listen()

        while True:
            tcp_connection, address = s.accept()
            print(f"Connected with {address}")

            with tcp_connection:
                # Python cannot guarantee a minimum number of received bytes, so we do it manually
                image_size_bytes = b''
                while len(image_size_bytes) < 4:
                    image_size_bytes = image_size_bytes + tcp_connection.recv(4 - len(image_size_bytes))

                image_size = int.from_bytes(image_size_bytes[:4], byteorder="big")
                print(f"Number of bytes in image is {image_size}")

                image = b''
                while len(image) < image_size:
                    image = image + tcp_connection.recv(image_size - len(image))

                num_images_bytes = b''
                while len(num_images_bytes) < 4:
                    num_images_bytes = num_images_bytes + tcp_connection.recv(4 - len(num_images_bytes))
                num_images = int.from_bytes(num_images_bytes[:4], byteorder="big")

                feature_vector = process_image(image, preprocess, model)

                nearest_neighbours = client.nns(schema_name, entity_name, feature_vector, limit=num_images, vector_col='feature_vector', id_col='image_path')

                print(nearest_neighbours)
                print(type(nearest_neighbours))

                for neighbor in nearest_neighbours:
                    image_path = neighbor['image_path']
                    with open(image_path, 'rb') as fd:
                        file_array = bytearray(fd.read())
                        print(f"Array size measured at {len(file_array)}")
                        int_size = tcp_connection.send(len(file_array).to_bytes(4, byteorder="big"))
                        print(f"Sent {int_size} bytes for int")
                        file_size_sent = tcp_connection.send(file_array)
                        print(f"Sent {file_size_sent} bytes for file")


# Calculates the feature vector of an image in byte array form
def process_image(image_bytes, preprocess, model):
    image = preprocess(Image.open(io.BytesIO(image_bytes))).unsqueeze(0).to(device)
    [feature_vector] = model.encode_image(image).tolist()
    return feature_vector


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script acts as a server to connect to applications wanting to do a similarity search for an image')
    parser.add_argument('-p', '--port', type=int, help='The port used to accept new connections')

    args = parser.parse_args()
    main(args.port)