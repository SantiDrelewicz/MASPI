import numpy as np
from helper_functions import conv2

def cnnPool(poolDim, convolvedFeatures):
    """
    cnnPool Pools los descriptores provenientes de la convolucion
    La funcion usa el Pool PROMEDIO.

    Parametros:
    -----------
    poolDim 
        dimension de la region de pooling
    convolvedFeatures
        los descriptores a realizar el pool
        convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
    Devuelve:
    pooledFeatures 
        matriz de los features agrupados
        pooledFeatures(poolRow, poolCol, featureNum, imageNum)
    """

    numImages = convolvedFeatures.shape[3]
    numFilters = convolvedFeatures.shape[2]
    convolvedDim = convolvedFeatures.shape[0] # me sirve usar uno ya que las imagenes son cuadradas

    px, py = int(convolvedDim / poolDim), int(convolvedDim / poolDim)
    pooledFeatures = np.zeros((px, py, numFilters, numImages))

#% Instrucciones:
#%   Realizar el pool de los features en regiones de tamaño poolDim x poolDim,
#%   para obtener la matriz pooledFeatures de 
#%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 

#%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) es el valor 
#%   del descriptor featureNum de la imagen imageNum agrupada sobre la 
#%   region (poolRow, poolCol). 
#%   

#%%% IMPLEMENTAR AQUI %%%
    for imageNum in range(numImages):
        for filterNum in range(numFilters):
            #% Pooling simple de una imagen con un filtro
            pooledImage = np.zeros((px, py))

            img = convolvedFeatures[:,:,filterNum,imageNum]
            img = np.squeeze(np.squeeze(img))

            f = np.ones((poolDim, poolDim))/poolDim**2

            # Making a pool downsampling is equivalent to make a convolution
            # with all weigth equal to 1 and divided by the poolDim**2
            # with stride pool image
            stride = poolDim
            pooledImage += conv2(img, f, mode="valid")[::stride, ::stride]
            pooledFeatures[:, :, filterNum, imageNum] = pooledImage

    return pooledFeatures

