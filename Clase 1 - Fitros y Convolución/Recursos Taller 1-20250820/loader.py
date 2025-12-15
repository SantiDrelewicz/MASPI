# Cargar y transformar la imagen
from torchvision.transforms import transforms
from PIL import Image

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convertir a escala de grises
        transforms.ToTensor()    # Convertir a tensor
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Añadir un batch dimension
    return image