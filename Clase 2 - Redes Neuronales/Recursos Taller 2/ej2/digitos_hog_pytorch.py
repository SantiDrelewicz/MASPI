from sklearn.model_selection import train_test_split
import numpy as np
from torch import nn
from torch import optim
import torch
import os
import cv2
from skimage.feature import hog
import requests
import tarfile
import io


DIGITOS = ['0','1','2','3','4','5','6','7','8','9']
# PARCHE PARAMETROS
W_PARCHE = 12
H_PARCHE = 24
## HOG PARAMETROS
HOG_PIX_CELL        = 4
HOG_CELL_BLOCK      = 2
HOG_ORIENTATIONS    = 8
HOG_FEATURE_LENGTH  = 320


torch.manual_seed(0) # para tener siempre los mismos pesos iniciales


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        # Inicializacion de pesos capa 1
        nn.init.xavier_uniform_(self.layer1.weight)
        # pongo a cero los bias
        self.layer1.bias.data.fill_(0)
        # Inicializacion de pesos capa 2
        nn.init.xavier_uniform_(self.layer2.weight)
        # pongo a cero los bias
        self.layer2.bias.data.fill_(0)

    def forward(self, input):
        out = self.layer1(input)
        out = torch.sigmoid(out)
        out = self.layer2(out)
        return torch.sigmoid(out)
    

def download_images(download_path: str):
    """
    Descarga el archivo .tar.gz desde la URL y extrae únicamente 
    la carpeta 'BaseOCR_MultiStyle' directamente en el directorio
    donde está este script.
    """
    url = "https://www.ipol.im/pub/art/2018/173/svmsmo_1.tar.gz"
    target_folder = "svm_smo/SVMCode/Datasets/BaseOCR_MultiStyle"
    
    # Descargar en memoria
    response = requests.get(url)
    response.raise_for_status()
    
    file_like = io.BytesIO(response.content)
    
    with tarfile.open(fileobj=file_like, mode="r:gz") as tar:
        # Filtrar miembros de la carpeta deseada
        members = [m for m in tar.getmembers() 
                   if m.name.startswith(target_folder + "/")]
        
        if not members:
            raise ValueError(
                f"La carpeta '{target_folder}' no se encontró en el archivo."
            )
        
        # Reescribir los nombres para que arranquen desde 'BaseOCR_MultiStyle'
        for m in members:
            # Quitar el prefijo "svmsmo_1/svm_smo/SVMCode/Datasets/"
            m.name = m.name.replace(target_folder, "BaseOCR_MultiStyle", 1)
        
        # Extraer directamente en el directorio del script
        tar.extractall(path=download_path, members=members)


def load_dataset(imgs_folder: str):
    # carpeta donde esta este script
    Nc = len(DIGITOS)    
    data, target = [], []
    for d in DIGITOS:
        pics = os.listdir(os.path.join(imgs_folder, d))
        for pic in pics:
            try:
                img_path = os.path.join(imgs_folder, d, pic)
                img = cv2.imdecode(
                    np.fromfile(img_path, dtype=np.uint8), 
                    cv2.IMREAD_GRAYSCALE
                )
                if len(img.shape) == 3: # only grayscale images
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                fd = hog(
                    cv2.resize(img, (W_PARCHE, H_PARCHE)), 
                    orientations=HOG_ORIENTATIONS, 
                    pixels_per_cell=(HOG_PIX_CELL, HOG_PIX_CELL), 
                    cells_per_block=(HOG_CELL_BLOCK, HOG_CELL_BLOCK)
                )
            except:
                pass
            else:
                data.append(fd)
                v = np.zeros((Nc))
                v[int(d)] = 1.
                target.append(v)
    return {
        "data": data, "target": target
    }


def train(
    model, 
    loss_fnc, optimizer, 
    X_train, y_train, 
    X_test, y_test,
    nepochs=10
):
    for epoch in range(nepochs):
        running_loss = 0.
        running_correct = 0.
        model.train()
        for i, [x,t] in enumerate(zip(X_train, y_train)):
            x = torch.tensor(x)
            t = torch.tensor(t)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fnc(output, t)
            loss.backward()
            optimizer.step()
            # estadisticos
            running_loss += loss.item()
            _,preds = torch.max(output,0)
            _,labels = torch.max(t,0)
            running_correct += torch.sum(preds == labels)

        epoch_loss = running_loss / (i-1)
        epoch_correct = running_correct / (i-1)

        running_loss = 0.
        running_correct = 0.
        model.eval()
        for j, [x,t] in enumerate(zip(X_test, y_test)):
            x = torch.tensor(x)
            t = torch.tensor(t)
            with torch.no_grad():
                output = model(x)
                # estadisticos
                _,preds = torch.max(output,0)
                _,labels = torch.max(t,0)
                running_correct += torch.sum(preds == labels)

        val_correct = running_correct / (j-1)    
        print('Epoca: %d - train loss: %f - train correctos: %f - test correctos: %f' % \
              (epoch, epoch_loss, epoch_correct,val_correct))


if __name__ == "__main__":
    dataset_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "datos"
    )
    if not os.path.exists(dataset_folder):
        download_images(dataset_folder)
    img_dataset = load_dataset(
        os.path.join(dataset_folder, "BaseOCR_MultiStyle")
    )
    # declaracion modelo
    Nh = 18
    model = MLP(
        input_size=HOG_FEATURE_LENGTH, 
        hidden_size=Nh, 
        output_size=len(DIGITOS)
    )
    loss_fnc = nn.MSELoss()
    optimizer =  optim.SGD(model.parameters(), lr=1e-1)
    test_size = 0.25
    nepochs = 50
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(img_dataset["data"], dtype=np.float32), 
        np.array(img_dataset["target"], dtype=np.float32), 
        test_size=test_size, random_state=42
    )
    train(
        model, loss_fnc, optimizer, 
        X_train, y_train, 
        X_test, y_test,
        nepochs=nepochs
    )
