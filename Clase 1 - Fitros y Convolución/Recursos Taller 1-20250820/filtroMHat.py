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
def mexican_hat(kernel_size=21, sigma=3.0):
    """
    Genera un filtro Mexican Hat (Laplacian of Gaussian) como tensor de Torch.
    
    Parámetros:
        kernel_size (int): tamaño del kernel (impar recomendado).
        sigma (float): desviación estándar de la gaussiana.
    
    Retorna:
        torch.Tensor: filtro 2D normalizado.
    """
    # Crear coordenadas
    ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    
    # Fórmula del Mexican Hat (LoG)
    r2 = xx**2 + yy**2
    norm = 1 / (torch.pi * sigma**4)
    kernel = norm * (1 - r2 / (2 * sigma**2)) * torch.exp(-r2 / (2 * sigma**2))
    
    # Normalizar para que la suma sea cero (característica de LoG)
    kernel -= kernel.mean()
    
    return kernel.unsqueeze(0).unsqueeze(0)

# Aplicar el filtro Gaussiano
def apply_MH_filter(image, kernel):
    return F.conv2d(image, kernel, padding=kernel.size(-1) // 2)


if __name__ == "__main__":
    # Parámetros del filtro Gaussiano
    kernel_size = 21  # Tamaño del kernel
    sigma = 5      # Desviación estándar

    # Cargar imagen
    image_path = 'fig2.jpg'
    image = load_image(image_path)

    # Crear y aplicar el filtro Gaussiano
    mh_kernel = mexican_hat(kernel_size, sigma)
    blurred_image = apply_MH_filter(image, mh_kernel)

    x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    y = x.clone()
    X, Y = torch.meshgrid(x, y, indexing='xy')
    Z = mh_kernel.squeeze()

    #imprimir filtro
    # print(mh_kernel)

    fig = plt.figure(figsize=(15, 5))

    # Imagen original
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title('Imagen Original')
    ax1.imshow(image.squeeze(), cmap='gray')

    # Imagen filtrada
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title('Imagen con Filtro Mexican Hat')
    ax2.imshow(blurred_image.squeeze().detach().numpy(), cmap='gray')

    # Kernel 3D
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), cmap='viridis')
    ax3.set_title('Kernel Mexican Hat')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Valor')

    plt.show()






