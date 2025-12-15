import numpy as np
from helper_functions import conv2
from scipy.special import expit


def cnnConvolve(filterDim: int, numFilters: int, 
                images: np.ndarray, 
                Wc: np.ndarray, bc: np.ndarray) -> np.ndarray:
    """
    cnnConvolve Devuelve el resultado de hacer la convolucion de W y b con
    las imagenes de entrada.

    Parametros:
    -----------
    - filterDim: dimension del filtro
    - numFilters: cantidad de filtros
    - images: imagenes 2D para convolucionar. Estas imagenes tienen un solo
        canal (gray scaled). El array images es del tipo images(r, c, image number)
    - Wc, bc: Wc, bc para calcular los features
        - Wc tiene tamanio (filterDim,filterDim,numFilters)
        - bc tiene tamanio (numFilters,1)    
    Devuelve:
    ---------
    - convolvedFeatures: matriz de descriptores convolucionados de la forma
                         convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
    """
    
    imageDim, numImages = images.shape[0], images.shape[2]
    convDim = imageDim - filterDim + 1
    convolvedFeatures = np.zeros((convDim, convDim, numFilters, numImages))
    
    #% Instrucciones:
    #%   Convolucionar cada filtro con cada imagen para obtener un array  de
    #%   tamaño (imageDim - filterDim + 1) x (imageDim - filterDim + 1) x numFeatures x numImages
    #%   de modo que convolvedFeatures(imageRow, imageCol, featureNum, imageNum) 
    #%   es el valor del descriptor featureNum para la imagen imageNum en la
    #%   region (imageRow, imageCol) to (imageRow + filterDim - 1, imageCol + filterDim - 1)

    for imageNum in range(numImages):
        for filterNum in range(numFilters):
            #% convolucion simple de una imagen con un filtro
            convolvedImage = np.zeros((convDim, convDim), dtype=np.float32)
            #% Obtener el filtro (filterDim x filterDim) 
            f = Wc[:,:,filterNum]
    
            #% Girar la matriz dada la definicion de convolucion
            f = np.rot90(np.squeeze(f), 2)
            #% Obtener la imagen
            im = np.squeeze(images[:, :, imageNum])
    
            #%%% IMPLEMENTACION AQUI %%%
            #% Convolucionar "filter" con "im", y adicionarlos a convolvedImage 
            #% para estar seguro de realizar una convolucion 'valida';
            #% Girar la matriz dada la definicion de convolucion si es necesario (con conv2 no lo es).
            f = np.rot90(f, 2)
            convolvedImage += conv2(im, f, mode='valid')
            #%%% IMPLEMENTACION AQUI %%%
            #% Agregar el bias 
            convolvedImage += bc[filterNum]
            #%%% IMPLEMENTACION AQUI %%%
            #% Luego, aplicar la funcion sigmoide para obtener la activacion de 
            #% la neurona.
            convolvedFeatures[:, :, filterNum, imageNum] = expit(convolvedImage)

    return convolvedFeatures