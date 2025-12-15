#  Dada la imagen de entrada fig3.jpg, determinar el filtro (entre k1, k2 y k3) 
#  y el umbral utilizado tal que el resultado sea lo más similar posible 
#  a la imagen binaria fig3-filtrada.png
#
#  Sugerencia: analizar primero cual es el filtro conveniente segun el resultado deseado
#  y luego determinar el umbral.
#

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Cargar y transformar la imagen
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convertir a escala de grises
        transforms.ToTensor()    # Convertir a tensor
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Añadir un batch dimension
    return image

# Filtros
k1 = torch.tensor(
    [[ 1,  1,  1], 
     [ 1, -2,  1], 
     [-1, -1, -1]], dtype=torch.float32
).unsqueeze(0).unsqueeze(0)
k2 = torch.tensor(
    [[3,  3,  3], 
     [3,  0, -5], 
     [3, -5, -5]], dtype=torch.float32
).unsqueeze(0).unsqueeze(0)
k3 = torch.tensor(
    [[-5, 3, 3], 
     [-5, 0, 3], 
     [-5, 3, 3]], dtype=torch.float32
).unsqueeze(0).unsqueeze(0)

# Aplicar el filtro de convolución
def aplicar_filtro(image, kernel):
    return F.conv2d(image, kernel, padding=1)


def aplicar_umbral(image, umbral):
    # Función que dada una imagen de entrada (tensor) aplica el umbral retornando
    # una imagen binaria, con ceros en la pixeles que no superan el valor del umbral.
    thr_layer = torch.nn.Threshold(threshold=umbral, value=0)
    return thr_layer(image) 


if __name__ == "__main__":
    # Cargar imagen
    image_path = 'fig3.jpg'
    image = load_image(image_path)
    print(image.shape)

    # Aplicar filtro
    imgfiltrada = aplicar_filtro(image, k2)

    # Encontrar el mejor umbral!
    thr_val = 4.5

    # Aplicar umbral
    imgthr = aplicar_umbral(torch.abs(imgfiltrada), thr_val)
    #imgthr = imgfiltrada

    # Visualizar resultados
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title('Imagen Original')
    plt.imshow(image.squeeze(), cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Imagen filtrada')
    plt.imshow(imgfiltrada.squeeze().detach().numpy(), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Imagen de umbral')
    plt.imshow(imgthr.squeeze().detach().numpy(), cmap='gray')

    plt.show()