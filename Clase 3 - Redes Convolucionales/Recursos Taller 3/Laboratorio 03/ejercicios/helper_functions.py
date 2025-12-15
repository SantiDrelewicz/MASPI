import numpy as np
from scipy.signal import convolve as conv2
import cv2, os


def loadDataset(dataset_folder, digitos, Nc, imageDim, rm_dataset_folder=False):
    data, target = [], []
    for d in digitos:
        pics = os.listdir(os.path.join(dataset_folder, d))
        for pic in pics:
            try:
                img_path = os.path.join(dataset_folder, d, pic)
                img = cv2.imdecode(
                    np.fromfile(img_path, dtype=np.uint8), 
                    cv2.IMREAD_GRAYSCALE
                )
            except:
                print('Problema con picture ', img_path)
                continue
            if img is None:
                print("No se pudo leer ", img_path)
                continue
            elif len(img.shape) == 3: # only grayscale images
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (imageDim, imageDim))
            data.append(img)
            v = np.zeros((Nc))
            v[int(d)] = 1.
            target.append(v)
    # Se devuelve traspuesta ya que el primer axis es el nro de samples            
    data, target = np.array(data, dtype=np.float32).T, np.array(target, dtype=np.float32).T
    if rm_dataset_folder:
        import shutil
        shutil.rmtree(dataset_folder)
    return data, target


def cnnInitParams(
    imageDim: int, filterDim: int, numFilters: int, poolDim: int, numClasses: int
):
    """
    Initialize parameters for a single layer convolutional neural
    network followed by a softmax layer.  

    Parameters
    -----------
    imageDim   :  height/width of image
    filterDim  :  dimension of convolutional filter                            
    numFilters :  number of convolutional filters
    poolDim    :  dimension of pooling area
    numClasses :  number of classes to predict

    Returns
    -------
    theta : unrolled parameter vector with initialized weights
    """
    #% Initialize parameters randomly based on layer sizes.
    assert(filterDim < imageDim),'filterDim must be lower than imageDim'
    Wc = 1e-1*np.random.randn(filterDim,filterDim,numFilters)

    outDim = imageDim - filterDim + 1 #% dimension of convolved image
    #% assume outDim is multiple of poolDim
    assert(outDim % poolDim==0) ,'poolDim must divide imageDim - filterDim + 1'
    outDim = outDim // poolDim
    hiddenSize = numFilters * outDim**2
    #% we'll choose weights uniformly from the interval [-r, r]
    r  = np.sqrt(6) / np.sqrt(numClasses+hiddenSize+1)
    Wd = 2*r*np.random.rand(numClasses, hiddenSize) - r
    bc = np.zeros((numFilters, 1))
    bd = np.zeros((numClasses, 1))
    # % Convert weights and bias gradients to the vector form.
    # % This step will "unroll" (flatten and concatenate together) all 
    # % your parameters into a vector, which can then be used with minFunc. 
    theta = np.vstack((
        np.expand_dims(Wc.flatten(), 1), np.expand_dims(Wd.flatten(), 1), 
        np.expand_dims(bc.flatten(), 1), np.expand_dims(bd.flatten(), 1)
    ))
    return theta


def cnnParamsToStack(
    theta: np.ndarray, 
    imageDim: int, 
    filterDim: int, numFilters: int, 
    poolDim: int, 
    numClasses: int
):
    """
    Converts unrolled parameters for a single layer convolutional neural
    network followed by a softmax layer into structured weight
    tensors/matrices and corresponding biases

    Parameters
    ----------
    theta : unrolled parameter vector
    imageDim : height/width of image
    filterDim : dimension of convolutional filter
    numFilters : number of convolutional filters
    poolDim : dimension of pooling area
    numClasses : number of classes to predict

    Returns
    -------
    Wc : filterDim x filterDim x numFilters parameter matrix
    Wd : numClasses x hiddenSize parameter matrix, hiddenSize is
         calculated as numFilters*((imageDim-filterDim+1)/poolDim)^2
    bc : bias for convolution layer of size numFilters x 1
    bd : bias for dense layer of size hiddenSize x 1
    """
    outDim = (imageDim - filterDim + 1) // poolDim
    hiddenSize = numFilters * outDim**2

    #%% Reshape theta
    indS = 0
    indE = numFilters * filterDim**2
    Wc = np.reshape(theta[indS:indE], (filterDim, filterDim, numFilters))
    indS = indE
    indE = indE + hiddenSize*numClasses
    Wd = np.reshape(theta[indS:indE], (numClasses, hiddenSize))
    indS = indE
    indE = indE + numFilters
    bc = theta[indS:indE]
    bd = theta[indE:]

    return Wc, Wd, bc, bd


