# -*- coding: utf-8 -*-
# %pip install -q torch_snippets
import torch
import modelo
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch_snippets import *

################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
################################################################
def train_batch(input, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, input)
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def validate_batch(input, model, criterion):
    model.eval()
    output = model(input)
    loss = criterion(output, input)
    return loss
################################################################
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Lambda(lambda x: x.to(device))
])
# Usar el dataset LPR
# Instanciar la clase Dataset que cargue los datos
trn_ds = Dataset()
val_ds = Dataset()
# Instanciar la clase Dataloader que sea usada luego para el entrenamiento
trn_dl = DataLoader()
val_dl = DataLoader()
# y aplique a cada imagen la transformacion img_transform
model = modelo.AutoEncoder(10).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

num_epochs = 5

train_loss = []
val_loss = []
for epoch in range(num_epochs):
    N = len(trn_dl)
    tloss = 0.
    for ix, (data, _) in enumerate(trn_dl):
        loss = train_batch(data, model, criterion, optimizer)
        tloss += loss.item()
    train_loss.append(tloss / N)
    N = len(val_dl)
    vloss = 0.
    for ix, (data, _) in enumerate(val_dl):
        loss = validate_batch(data, model, criterion)
        vloss += loss.item()
    vloss.append(vloss / N)    

####################################################################
# graficar las losses de entrenamiento y de validacion

####################################################################
for _ in range(3):
    ix = np.random.randint(len(val_ds))
    im, _ = val_ds[ix]
    _im = model(im[None])[0]
    fig, ax = plt.subplots(1,2,figsize=(3,3)) 
    show(im[0], ax=ax[0], title='input')
    show(_im[0], ax=ax[1], title='prediction')
    plt.tight_layout()
    plt.show()