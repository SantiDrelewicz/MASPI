import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from loader import load_image

#windows
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#set KMP_DUPLICATE_LIB_OK=True
#linux
#export KMP_DUPLICATE_LIB_OK=True

# Crear el filtro Gaussiano
def gaussian_kernel(size, sigma):
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    x = x.repeat(size, 1)
    y = x.t()
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalizar el kernel
    return kernel.unsqueeze(0).unsqueeze(0)

# Aplicar el filtro Gaussiano
def apply_gaussian_filter(image, kernel):
    return F.conv2d(image, kernel, padding=kernel.size(-1) // 2)


if __name__ == "__main__":
    # Parámetros del filtro Gaussiano
    kernel_size = 81  # Tamaño del kernel
    sigma = 3      # Desviación estándar

    # Cargar imagen
    image_path = 'fig2.jpg'
    image = load_image(image_path)

    # Crear y aplicar el filtro Gaussiano
    gauss_kernel = gaussian_kernel(kernel_size, sigma)
    blurred_image = apply_gaussian_filter(image, gauss_kernel)

    x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    y = x.clone()
    X, Y = torch.meshgrid(x, y, indexing='xy')
    Z = gauss_kernel.squeeze()

    #imprimir filtro
    # print(gaussian_kernel)

    fig = plt.figure(figsize=(15, 5))

    # Imagen original
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title('Imagen Original')
    ax1.imshow(image.squeeze(), cmap='gray')

    # Imagen filtrada
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title('Imagen con Filtro Gaussiano')
    ax2.imshow(blurred_image.squeeze().detach().numpy(), cmap='gray')

    # Kernel 3D
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), cmap='viridis')
    ax3.set_title('Kernel Gaussiano')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Valor')

    plt.show()





