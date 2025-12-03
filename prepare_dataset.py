import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import os
import shutil
from tqdm import tqdm

print("="*60)
print("PREPARACIÓN DEL DATASET CIFAR-10 (4 CLASES)")
print("="*60)

# Definir las 4 clases seleccionadas de CIFAR-10
SELECTED_CLASSES = {
    0: 'airplane',
    1: 'automobile',
    8: 'ship',
    9: 'truck'
}

CLASS_NAMES = ['airplane', 'automobile', 'ship', 'truck']

# Crear directorios necesarios
os.makedirs('data', exist_ok=True)
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/test', exist_ok=True)
os.makedirs('data/sample_images', exist_ok=True)

print("\n[1/5] Descargando CIFAR-10 dataset completo...")
# Descargar CIFAR-10 completo (50,000 imágenes de entrenamiento)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Descargar el dataset completo
full_dataset = datasets.CIFAR10(root='./data/cifar10_original', train=True, 
                                download=True, transform=None)

print(f"Dataset completo descargado: {len(full_dataset)} imágenes")

print("\n[2/5] Filtrando solo las 4 clases seleccionadas...")
# Filtrar solo las imágenes de las 4 clases seleccionadas
selected_indices = []
for idx in tqdm(range(len(full_dataset)), desc="Filtrando clases"):
    _, label = full_dataset[idx]
    if label in SELECTED_CLASSES.keys():
        selected_indices.append(idx)

selected_indices = np.array(selected_indices)
print(f"\nTotal de imágenes seleccionadas: {len(selected_indices)}")
print("Distribución por clase:")
for class_id, class_name in SELECTED_CLASSES.items():
    count = sum([1 for idx in selected_indices if full_dataset[idx][1] == class_id])
    print(f"  - {class_name}: {count} imágenes")

print("\n[3/5] Dividiendo en train (80%) y test (20%)...")
# Dividir en train (80%) y test (20%)
np.random.seed(42)  # Para reproducibilidad
np.random.shuffle(selected_indices)
split_point = int(0.8 * len(selected_indices))
train_indices = selected_indices[:split_point]
test_indices = selected_indices[split_point:]

print(f"Train: {len(train_indices)} imágenes")
print(f"Test: {len(test_indices)} imágenes")

# Función para guardar imágenes
def save_images(indices, folder_name):
    """Guarda imágenes en el folder especificado"""
    print(f"\nGuardando imágenes en {folder_name}/...")
    
    for idx in tqdm(indices, desc=f"Guardando en {folder_name}"):
        img, label = full_dataset[idx]
        class_name = SELECTED_CLASSES[label]
        
        # Crear carpeta para la clase si no existe
        class_folder = os.path.join(folder_name, class_name)
        os.makedirs(class_folder, exist_ok=True)
        
        # Guardar imagen
        img_path = os.path.join(class_folder, f"{idx}.png")
        img.save(img_path)

print("\n[4/5] Guardando imágenes de entrenamiento...")
save_images(train_indices, 'data/train')

print("\n[5/5] Guardando imágenes de prueba...")
save_images(test_indices, 'data/test')

print("\n[6/6] Copiando 10 imágenes de cada clase para muestras...")
# Seleccionar 10 imágenes de cada clase del test set
samples_per_class = {}
for class_id in SELECTED_CLASSES.keys():
    samples_per_class[class_id] = []

# Agrupar índices del test por clase
for idx in test_indices:
    _, label = full_dataset[idx]
    if len(samples_per_class[label]) < 10:
        samples_per_class[label].append(idx)

# Guardar las muestras
total_samples = 0
for class_id, class_name in SELECTED_CLASSES.items():
    for idx in tqdm(samples_per_class[class_id], desc=f"Copiando {class_name}"):
        img, label = full_dataset[idx]
        
        # Guardar con nombre descriptivo
        sample_path = os.path.join('data/sample_images', f"{idx}_{class_name}.png")
        img.save(sample_path)
        total_samples += 1

print("\n" + "="*60)
print("RESUMEN DE LA PREPARACIÓN")
print("="*60)
print(f"✓ Dataset original: 50,000 imágenes")
print(f"✓ Clases seleccionadas: 4 (airplane, automobile, ship, truck)")
print(f"✓ Total de imágenes filtradas: {len(selected_indices)}")
print(f"✓ Train set: {len(train_indices)} imágenes (80%)")
print(f"✓ Test set: {len(test_indices)} imágenes (20%)")
print(f"✓ Muestras guardadas: {total_samples} imágenes (10 por clase)")
print("\nEstructura de directorios creada:")
print("  data/train/         - Imágenes de entrenamiento por clase")
print("  data/test/          - Imágenes de prueba por clase")
print("  data/sample_images/ - 40 imágenes de muestra (10 por clase)")
print("\n¡Preparación completada exitosamente!")
print("="*60)
