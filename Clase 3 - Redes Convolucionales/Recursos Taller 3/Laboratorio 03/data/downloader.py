import requests
import tarfile
import io


def downloadImages(download_path: str):
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
