import socket
from os.path import exists
from os import remove
from os import mkdir
from PIL import Image

HOST = "10.34.58.150"
PORT = 8990
NUM_IMAGES = 10
image = "./MySQL/bus.jpg"

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, open(image, 'rb') as f:
    s.connect((HOST, PORT))

    image_bytes = f.read()

    s.send(len(image_bytes).to_bytes(4, byteorder="big"))
    s.send(image_bytes)
    s.send(NUM_IMAGES.to_bytes(4, byteorder="big"))

    if not exists("tmp"):
        mkdir("tmp")

    for i in range(10):
        image_size_bytes = b''
        while len(image_size_bytes) < 4:
            image_size_bytes = image_size_bytes + s.recv(4 - len(image_size_bytes))
        size = int.from_bytes(image_size_bytes[:4], byteorder="big")

        print(f"size is {size}")

        file_data = b""

        while len(file_data) < size:
            data = s.recv(min(4096, size - len(file_data)))
            file_data += data
        print(f"length of array is {len(file_data)}")
        if exists(f"tmp/test{i}.jpg"):
            remove(f"tmp/test{i}.jpg")
        with open(f"tmp/test{i}.jpg", 'bx') as file:
            file.write(file_data)

