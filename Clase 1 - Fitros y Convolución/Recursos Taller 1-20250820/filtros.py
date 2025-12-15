import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from loader import load_image

# Definir filtros de Prewitt
prewitt_x = torch.tensor(
    [[1, 0, -1], 
     [1, 0, -1], 
     [1, 0, -1]], 
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0)

prewitt_y = torch.tensor(
    [[ 1,  1,  1], 
     [ 0,  0,  0], 
     [-1, -1, -1]], 
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0)

print(prewitt_x.shape)

# Aplicar el filtro de convolución
def apply_prewitt_filter(image, kernel):
    return F.conv2d(image, kernel, padding=1)


if __name__ == "__main__":
    # Cargar imagen
    for n in [1,2,3,4]:
        image_path = f'fig{n}.jpg'
        image = load_image(image_path)
        print(image.shape)

        # Aplicar filtros de Prewitt
        edges_x = apply_prewitt_filter(image, prewitt_x)
        edges_y = apply_prewitt_filter(image, prewitt_y)

        # Calcular la magnitud de los bordes
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)

        # Visualizar resultados
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 4, 1)
        plt.title('Imagen Original')
        plt.imshow(image.squeeze(), cmap='gray')

        plt.subplot(1, 4, 3)
        plt.title('Filtro Prewitt Horizontal')
        plt.imshow(edges_x.squeeze().detach().numpy(), cmap='gray')

        plt.subplot(1, 4, 4)
        plt.title('Filtro Prewitt Vertical')
        plt.imshow(edges_y.squeeze().detach().numpy(), cmap='gray')

        plt.subplot(1, 4, 2)
        plt.title('Modulo de los gradientes')
        plt.imshow(edges.squeeze().detach().numpy(), cmap='gray')

        plt.show()