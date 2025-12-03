import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# Añadir el directorio actual al path para importar mlp_model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlp_model import MLP

print("="*70)
print("ENTRENAMIENTO DEL MODELO MLP - CIFAR-10 (4 CLASES)")
print("="*70)

# Configuración de hiperparámetros optimizados
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
DROPOUT_RATE = 0.4
EARLY_STOPPING_PATIENCE = 8

INPUT_SIZE = 3072  # 32x32x3
HIDDEN_SIZES = [512, 256, 128]
NUM_CLASSES = 4

# Clases del dataset
CLASS_NAMES = ['airplane', 'automobile', 'ship', 'truck']

# Crear directorio para guardar plots
os.makedirs('../plots', exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[INFO] Usando dispositivo: {device}")
if device.type == 'cpu':
    print("[WARNING] Entrenamiento en CPU - esto puede ser lento")

class CIFAR10CustomDataset(Dataset):
    """Dataset personalizado para cargar imágenes desde carpetas"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Mapeo de nombres de clase a índices
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        
        # Cargar todas las imágenes
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.png') or img_name.endswith('.jpg'):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Transformaciones para entrenamiento (con data augmentation)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Transformaciones para test (sin augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("\n[1/6] Cargando datasets...")
train_dataset = CIFAR10CustomDataset(root_dir='../data/train', transform=train_transform)
test_dataset = CIFAR10CustomDataset(root_dir='../data/test', transform=test_transform)

print(f"✓ Train dataset: {len(train_dataset)} imágenes")
print(f"✓ Test dataset: {len(test_dataset)} imágenes")

# Distribución por clase
train_class_counts = {}
for _, label in train_dataset:
    class_name = CLASS_NAMES[label]
    train_class_counts[class_name] = train_class_counts.get(class_name, 0) + 1

print("\nDistribución de clases en train:")
for class_name in CLASS_NAMES:
    count = train_class_counts.get(class_name, 0)
    print(f"  - {class_name}: {count} imágenes")

# Data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, 
                         shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, 
                        shuffle=False, num_workers=0)

print(f"\n[2/6] Inicializando modelo MLP mejorado...")
model = MLP(input_size=INPUT_SIZE, hidden_sizes=HIDDEN_SIZES, 
           num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE).to(device)

# Información del modelo
model_info = model.get_model_info()
print(f"✓ Arquitectura: {model_info['architecture']}")
print(f"✓ Parámetros totales: {model_info['total_parameters']:,}")
print(f"✓ Parámetros entrenables: {model_info['trainable_parameters']:,}")

# Loss con label smoothing y optimizer AdamW con weight decay
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

print(f"\n[3/6] Configuración de entrenamiento:")
print(f"✓ Optimizer: AdamW (weight_decay={WEIGHT_DECAY})")
print(f"✓ Learning rate: {LEARNING_RATE}")
print(f"✓ LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")
print(f"✓ Batch size: {BATCH_SIZE}")
print(f"✓ Epochs: {NUM_EPOCHS}")
print(f"✓ Early stopping patience: {EARLY_STOPPING_PATIENCE}")
print(f"✓ Loss function: CrossEntropyLoss (label_smoothing={LABEL_SMOOTHING})")
print(f"✓ Dropout rate: {DROPOUT_RATE}")
print(f"✓ Data augmentation: RandomFlip, RandomCrop, ColorJitter, RandomRotation")

# Listas para guardar métricas
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
learning_rates = []

# Variables para early stopping
best_test_acc = 0.0
best_test_loss = float('inf')
patience_counter = 0
best_epoch = 0

def evaluate_model(model, data_loader, criterion, device):
    """Evalúa el modelo en un conjunto de datos"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

print("\n" + "="*70)
print("[4/6] INICIANDO ENTRENAMIENTO")
print("="*70)

start_time = datetime.now()

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Barra de progreso para el epoch
    pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Estadísticas
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Actualizar barra de progreso
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.6f}'})
    
    # Calcular métricas de entrenamiento
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    
    # Evaluar en test set
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    
    # Guardar métricas
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    # Guardar learning rate actual
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    
    # Actualizar learning rate scheduler basado en test loss
    scheduler.step(test_loss)
    
    # Imprimir resultados del epoch
    print(f'\nEpoch [{epoch+1}/{NUM_EPOCHS}]:')
    print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}%')
    print(f'  Test Loss:  {test_loss:.4f} | Test Acc:  {test_accuracy:.2f}%')
    print(f'  Learning Rate: {current_lr:.6f}')
    
    # Early stopping basado en test accuracy
    if test_accuracy > best_test_acc:
        best_test_acc = test_accuracy
        best_test_loss = test_loss
        best_epoch = epoch + 1
        patience_counter = 0
        # Guardar el mejor modelo
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
        }, 'best_model.pth')
        print(f'  ✓ Mejor modelo guardado! (Test Acc: {best_test_acc:.2f}%)')
    else:
        patience_counter += 1
        print(f'  Early stopping counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}')
    
    print('-'*70)
    
    # Early stopping
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f'\n[EARLY STOPPING] No mejora en {EARLY_STOPPING_PATIENCE} epochs.')
        print(f'Mejor modelo en epoch {best_epoch} con Test Acc: {best_test_acc:.2f}%')
        break

end_time = datetime.now()
training_duration = (end_time - start_time).total_seconds()

print("\n" + "="*70)
print("[5/6] GUARDANDO MODELO Y MÉTRICAS")
print("="*70)

# Guardar el modelo final
model_path = 'cifar10_mlp.pth'
torch.save({
    'epoch': len(train_losses),
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_losses[-1],
    'test_loss': test_losses[-1],
    'train_accuracy': train_accuracies[-1],
    'test_accuracy': test_accuracies[-1],
    'best_test_accuracy': best_test_acc,
    'best_epoch': best_epoch,
    'class_names': CLASS_NAMES,
    'input_size': INPUT_SIZE,
    'hidden_sizes': HIDDEN_SIZES,
    'num_classes': NUM_CLASSES,
    'dropout_rate': DROPOUT_RATE,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
}, model_path)

print(f"✓ Modelo guardado en: {model_path}")

# Guardar métricas en JSON
metrics = {
    'training_info': {
        'epochs_completed': len(train_losses),
        'total_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'label_smoothing': LABEL_SMOOTHING,
        'dropout_rate': DROPOUT_RATE,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'optimizer': 'AdamW',
        'device': str(device),
        'training_duration_seconds': training_duration,
        'training_duration_formatted': str(end_time - start_time),
        'early_stopped': patience_counter >= EARLY_STOPPING_PATIENCE
    },
    'model_info': model_info,
    'final_metrics': {
        'train_loss': train_losses[-1],
        'train_accuracy': train_accuracies[-1],
        'test_loss': test_losses[-1],
        'test_accuracy': test_accuracies[-1]
    },
    'best_metrics': {
        'best_test_accuracy': best_test_acc,
        'best_test_loss': best_test_loss,
        'best_epoch': best_epoch
    },
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'test_losses': test_losses,
    'test_accuracies': test_accuracies,
    'learning_rates': learning_rates,
    'class_names': CLASS_NAMES
}

with open('training_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"✓ Métricas guardadas en: training_metrics.json")

print("\n" + "="*70)
print("[6/6] GENERANDO PLOTS")
print("="*70)

# Plot 1: Loss curves
epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
plt.plot(epochs_range, test_losses, 'r-', label='Test Loss', linewidth=2)
plt.axvline(x=best_epoch, color='g', linestyle='--', linewidth=2, label='Best Model', alpha=0.7)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Test Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 2: Accuracy curves
plt.subplot(1, 3, 2)
plt.plot(epochs_range, train_accuracies, 'b-', label='Train Accuracy', linewidth=2)
plt.plot(epochs_range, test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
plt.axvline(x=best_epoch, color='g', linestyle='--', linewidth=2, label='Best Model', alpha=0.7)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Training and Test Accuracy', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 3: Learning Rate
plt.subplot(1, 3, 3)
plt.plot(epochs_range, learning_rates, 'g-', linewidth=2, marker='o', markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.title('Training and Test Accuracy', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/training_curves.png', dpi=300, bbox_inches='tight')
print("✓ Curvas de entrenamiento guardadas en: plots/training_curves.png")

# Plot 3: Comparación final
plt.figure(figsize=(10, 6))
metrics_names = ['Train Loss', 'Test Loss', 'Train Acc (%)', 'Test Acc (%)']
metrics_values = [train_losses[-1], test_losses[-1], train_accuracies[-1], test_accuracies[-1]]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Añadir valores en las barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel('Value', fontsize=12)
plt.title('Final Metrics Summary', fontsize=14, fontweight='bold')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/final_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Métricas finales guardadas en: plots/final_metrics.png")

print("\n" + "="*70)
print("RESUMEN FINAL")
print("="*70)
print(f"✓ Duración del entrenamiento: {training_duration/60:.2f} minutos")
print(f"✓ Epochs completados: {len(train_losses)}/{NUM_EPOCHS}")
if patience_counter >= EARLY_STOPPING_PATIENCE:
    print(f"✓ Early stopping activado en epoch {len(train_losses)}")
print(f"\n📊 MÉTRICAS FINALES:")
print(f"  Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accuracies[-1]:.2f}%")
print(f"  Test Loss:  {test_losses[-1]:.4f} | Test Acc:  {test_accuracies[-1]:.2f}%")
print(f"\n🏆 MEJOR MODELO:")
print(f"  Epoch: {best_epoch}")
print(f"  Test Accuracy: {best_test_acc:.2f}%")
print(f"  Test Loss: {best_test_loss:.4f}")
print(f"  Train-Test Gap: {train_accuracies[best_epoch-1] - best_test_acc:.2f}%")
print(f"\n✓ Modelo final guardado en: {model_path}")
print(f"✓ Mejor modelo guardado en: best_model.pth")
print("="*70)
print("¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
print("="*70)
