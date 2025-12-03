# 📚 Documentación Completa del Proyecto CIFAR-10 MLP Classifier

---

## 📋 Tabla de Contenidos
1. [Visión General del Proyecto](#visión-general-del-proyecto)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Estructura de Archivos](#estructura-de-archivos)
4. [Análisis Detallado de Cada Archivo](#análisis-detallado-de-cada-archivo)
5. [Flujo de Datos](#flujo-de-datos)
6. [Conceptos Importantes](#conceptos-importantes)
7. [Instalación y Uso](#instalación-y-uso)

---

## 🎯 Visión General del Proyecto

### Objetivo
Este proyecto implementa un clasificador de imágenes utilizando una **Red Neuronal Artificial (MLP - Multi-Layer Perceptron)** entrenada sobre el dataset CIFAR-10, específicamente para clasificar 4 categorías de imágenes:
- **Airplane** (Avión)
- **Automobile** (Automóvil)
- **Ship** (Barco)
- **Truck** (Camión)

### Tecnologías Principales
- **PyTorch**: Framework de deep learning para entrenar el modelo
- **FastAPI**: Backend API REST para servir predicciones
- **React**: Frontend web para interfaz de usuario
- **Docker**: Containerización para facilitar el despliegue

### Características Clave
- ✅ Modelo MLP simple pero efectivo (~68% accuracy en test)
- ✅ API REST para predicciones en tiempo real
- ✅ Interfaz web intuitiva con dos modos de clasificación
- ✅ Procesamiento automático de imágenes de cualquier tamaño
- ✅ Sistema completamente dockerizado

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                         USUARIO                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Frontend (React)      │
              │   Puerto: 3000          │
              │   - Carga de imágenes   │
              │   - Visualización       │
              │   - Resultados          │
              └──────────┬──────────────┘
                         │ HTTP POST /predict
                         ▼
              ┌────────────────────────┐
              │  Backend (FastAPI)      │
              │  Puerto: 8000           │
              │  - Recibe imagen        │
              │  - Preprocesa           │
              │  - Inferencia           │
              │  - Retorna resultado    │
              └──────────┬──────────────┘
                         │
                         ▼
              ┌────────────────────────┐
              │  Modelo MLP (PyTorch)   │
              │  cifar10_mlp.pth        │
              │  - 1.7M parámetros      │
              │  - 3 capas              │
              │  - 4 clases salida      │
              └─────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              DOCKER COMPOSE ORCHESTRATION                    │
│  - Red compartida: cifar10-network                          │
│  - Volumen compartido: ./model                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Estructura de Archivos

```
cifar10/
│
├── 📊 data/                              # Datos del dataset
│   ├── cifar10_original/                 # Dataset original descargado
│   │   └── cifar-10-batches-py/          # Archivos binarios de CIFAR-10
│   ├── train/                            # 19,200 imágenes de entrenamiento
│   │   ├── airplane/                     # ~4,800 imágenes
│   │   ├── automobile/                   # ~4,800 imágenes
│   │   ├── ship/                         # ~4,800 imágenes
│   │   └── truck/                        # ~4,800 imágenes
│   ├── test/                             # 4,800 imágenes de prueba
│   │   ├── airplane/
│   │   ├── automobile/
│   │   ├── ship/
│   │   └── truck/
│   └── sample_images/                    # 40 imágenes de muestra (10 por clase)
│
├── 🧠 model/                             # Modelo de red neuronal
│   ├── mlp_model.py                      # Definición de la arquitectura MLP
│   ├── train.py                          # Script de entrenamiento
│   ├── cifar10_mlp.pth                   # Modelo entrenado (pesos guardados)
│   └── training_metrics.json             # Métricas del entrenamiento
│
├── 🔧 backend/                           # API Backend
│   ├── main.py                           # Servidor FastAPI
│   ├── requirements.txt                  # Dependencias Python
│   └── Dockerfile                        # Containerización del backend
│
├── 🎨 frontend/                          # Interfaz Web
│   ├── src/
│   │   ├── App.js                        # Componente principal React
│   │   └── index.js                      # Punto de entrada
│   ├── public/
│   │   └── index.html                    # HTML base
│   ├── package.json                      # Dependencias Node.js
│   └── Dockerfile                        # Containerización del frontend
│
├── 📈 plots/                             # Gráficas del entrenamiento
│   ├── training_curves.png               # Loss y Accuracy por epoch
│   └── final_metrics.png                 # Resumen de métricas finales
│
├── 🐳 docker-compose.yml                 # Orquestación de contenedores
├── 🔨 prepare_dataset.py                 # Script para preparar el dataset
├── 📖 PLAN.md                            # Plan de implementación
└── 📚 DOCUMENTACION.md                   # Este archivo
```

---

## 🔍 Análisis Detallado de Cada Archivo

---

### 1️⃣ `prepare_dataset.py` - Preparación del Dataset

**Propósito**: Descarga, filtra y organiza el dataset CIFAR-10 para el entrenamiento.

#### 🔑 Funciones Principales

```python
# Descarga del dataset completo (50,000 imágenes)
full_dataset = datasets.CIFAR10(root='./data/cifar10_original', 
                                train=True, download=True)
```

**¿Qué hace?**
- Descarga automáticamente CIFAR-10 si no existe
- CIFAR-10 original tiene 10 clases, pero solo necesitamos 4

#### 📊 Proceso de Filtrado

```python
SELECTED_CLASSES = {
    0: 'airplane',
    1: 'automobile',
    8: 'ship',
    9: 'truck'
}
```

**¿Por qué estos índices?**
- CIFAR-10 usa índices específicos para cada clase
- Clase 0 = airplane, Clase 1 = automobile
- Clase 8 = ship, Clase 9 = truck
- Se filtran ~24,000 imágenes totales (~6,000 por clase)

#### 🔄 División de Datos

```python
split_point = int(0.8 * len(selected_indices))
train_indices = selected_indices[:split_point]  # 80%
test_indices = selected_indices[split_point:]   # 20%
```

**Concepto Importante: Train/Test Split**
- **Train (80%)**: 19,200 imágenes para entrenar el modelo
- **Test (20%)**: 4,800 imágenes para evaluar el rendimiento
- Esta división evita el **overfitting** (que el modelo memorice en vez de aprender)

#### 💾 Guardado de Imágenes

```python
def save_images(indices, folder_name):
    for idx in indices:
        img, label = full_dataset[idx]
        class_name = SELECTED_CLASSES[label]
        class_folder = os.path.join(folder_name, class_name)
        img_path = os.path.join(class_folder, f"{idx}.png")
        img.save(img_path)
```

**Organización**:
- Cada imagen se guarda en su carpeta de clase correspondiente
- Formato PNG para preservar calidad
- Nombres únicos usando el índice original

#### 🎯 Muestras de Prueba

```python
# 10 imágenes de cada clase -> sample_images/
samples_per_class[label].append(idx)
```

**Utilidad**:
- 40 imágenes totales para pruebas manuales rápidas
- Facilita verificar que el modelo funciona correctamente
- No se usan en entrenamiento ni evaluación

---

### 2️⃣ `model/mlp_model.py` - Arquitectura de la Red Neuronal

**Propósito**: Define la estructura del Multi-Layer Perceptron (MLP).

#### 🧠 Arquitectura del Modelo

```python
class MLP(nn.Module):
    def __init__(self, input_size=3072, hidden1=512, hidden2=256, num_classes=4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)    # 3072 -> 512
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)       # 512 -> 256
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes)   # 256 -> 4
```

**Desglose Capa por Capa**:

1. **Capa de Entrada (fc1)**:
   - **Input**: 3072 neuronas (32x32x3 = imagen RGB aplanada)
   - **Output**: 512 neuronas
   - **Parámetros**: 3072 × 512 + 512 (bias) = 1,573,376
   - **Función**: Extrae características básicas de la imagen

2. **Activación ReLU (relu1)**:
   ```python
   ReLU(x) = max(0, x)
   ```
   - **Propósito**: Introduce no-linealidad
   - **Sin ReLU**: La red solo podría aprender relaciones lineales
   - **Con ReLU**: Puede aprender patrones complejos

3. **Capa Oculta (fc2)**:
   - **Input**: 512 neuronas
   - **Output**: 256 neuronas
   - **Parámetros**: 512 × 256 + 256 = 131,328
   - **Función**: Combina características en representaciones más abstractas

4. **Activación ReLU (relu2)**: Otra capa de no-linealidad

5. **Capa de Salida (fc3)**:
   - **Input**: 256 neuronas
   - **Output**: 4 neuronas (una por clase)
   - **Parámetros**: 256 × 4 + 4 = 1,028
   - **Función**: Produce puntuaciones (logits) para cada clase

**Total de Parámetros**: 1,705,732

#### 🔄 Forward Pass

```python
def forward(self, x):
    x = x.view(x.size(0), -1)  # Aplanar imagen
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.fc2(x)
    x = self.relu2(x)
    x = self.fc3(x)
    return x
```

**Flujo de Datos**:
```
Imagen [batch, 3, 32, 32]
    ↓ view()
Vector [batch, 3072]
    ↓ fc1 + relu1
Vector [batch, 512]
    ↓ fc2 + relu2
Vector [batch, 256]
    ↓ fc3
Logits [batch, 4]
```

**Concepto Importante: Batch Processing**
- `batch` = número de imágenes procesadas simultáneamente
- Batch size de 64 significa 64 imágenes a la vez
- Más eficiente que procesar una por una

#### 📊 Información del Modelo

```python
def get_model_info(self):
    total_params = sum(p.numel() for p in self.parameters())
    trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
```

**¿Qué son los parámetros entrenables?**
- Pesos y biases que la red ajusta durante el entrenamiento
- En este modelo, todos los parámetros son entrenables
- Algunos modelos congelan capas (transfer learning)

---

### 3️⃣ `model/train.py` - Script de Entrenamiento

**Propósito**: Entrena el modelo MLP con los datos preparados.

#### ⚙️ Configuración de Hiperparámetros

```python
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 25
```

**Explicación de Hiperparámetros**:

- **BATCH_SIZE = 64**:
  - Procesa 64 imágenes simultáneamente
  - Trade-off: Más grande = más rápido pero más memoria
  - 64 es un valor estándar y eficiente

- **LEARNING_RATE = 0.001**:
  - Controla qué tan rápido aprende el modelo
  - Muy alto → el modelo puede no converger
  - Muy bajo → entrenamiento muy lento
  - 0.001 (1e-3) es un valor común para Adam

- **NUM_EPOCHS = 25**:
  - Una época = pasar por TODO el dataset una vez
  - 25 épocas = el modelo ve cada imagen 25 veces
  - Más épocas ≠ siempre mejor (riesgo de overfitting)

#### 📂 Dataset Personalizado

```python
class CIFAR10CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
```

**¿Por qué un Dataset personalizado?**
- PyTorch necesita saber cómo cargar nuestros datos
- Conecta archivos de imagen con sus etiquetas
- Aplica transformaciones automáticamente

#### 🔄 Transformaciones

```python
transform = transforms.Compose([
    transforms.ToTensor(),                           # PIL -> Tensor
    transforms.Normalize((0.5, 0.5, 0.5),           # Media RGB
                        (0.5, 0.5, 0.5))            # Std RGB
])
```

**¿Por qué normalizar?**
1. **ToTensor**: Convierte PIL Image a tensor PyTorch [0, 1]
2. **Normalize**: Escala a [-1, 1]
   ```
   normalized = (pixel - mean) / std
   normalized = (pixel - 0.5) / 0.5
   ```
3. **Beneficio**: Acelera el entrenamiento y mejora convergencia

#### 🎯 DataLoader

```python
train_loader = DataLoader(dataset=train_dataset, 
                         batch_size=BATCH_SIZE,
                         shuffle=True,           # Importante!
                         num_workers=0)
```

**Concepto Importante: Shuffle**
- **shuffle=True** en train: Aleatoriza el orden en cada época
- Previene que el modelo aprenda el orden de los datos
- **shuffle=False** en test: No necesario, solo evaluamos

#### 💪 Optimizador y Loss Function

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

**CrossEntropyLoss**:
- Función de pérdida para clasificación multi-clase
- Combina LogSoftmax + NLLLoss
- Penaliza predicciones incorrectas
- Fórmula: `Loss = -log(P(clase_correcta))`

**Adam Optimizer**:
- Algoritmo de optimización adaptativo
- Ajusta el learning rate automáticamente
- Mejor que SGD simple para muchos casos
- Combina momentum + RMSProp

#### 🔁 Loop de Entrenamiento

```python
for epoch in range(NUM_EPOCHS):
    model.train()  # Modo entrenamiento
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()  # Limpiar gradientes anteriores
        loss.backward()         # Calcular gradientes
        optimizer.step()        # Actualizar pesos
```

**Desglose del Training Loop**:

1. **model.train()**: Activa modo entrenamiento
   - Importante para Dropout y BatchNorm (no usados aquí)
   - Buena práctica siempre ponerlo

2. **Forward Pass**:
   ```python
   outputs = model(images)  # Predicciones
   loss = criterion(outputs, labels)  # Calcular error
   ```

3. **Backward Pass** (Backpropagation):
   ```python
   optimizer.zero_grad()  # Resetear gradientes
   loss.backward()         # Calcular ∂Loss/∂Weights
   optimizer.step()        # weights = weights - lr * gradient
   ```

**Concepto Crucial: Backpropagation**
- Calcula cómo cada peso contribuye al error
- Usa la regla de la cadena (cálculo)
- Permite ajustar pesos para reducir el error

#### 📊 Evaluación

```python
def evaluate_model(model, data_loader, criterion, device):
    model.eval()  # Modo evaluación
    with torch.no_grad():  # No calcular gradientes
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
```

**¿Por qué model.eval() y no_grad()?**
- **model.eval()**: Desactiva Dropout, BatchNorm en modo eval
- **torch.no_grad()**: Ahorra memoria, no necesitamos gradientes
- Combinados: Evaluación más rápida y precisa

#### 💾 Guardado del Modelo

```python
torch.save({
    'epoch': NUM_EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_losses[-1],
    'test_loss': test_losses[-1],
    'train_accuracy': train_accuracies[-1],
    'test_accuracy': test_accuracies[-1],
    'class_names': CLASS_NAMES,
    'input_size': INPUT_SIZE,
    'hidden1': HIDDEN1,
    'hidden2': HIDDEN2,
    'num_classes': NUM_CLASSES,
}, model_path)
```

**¿Qué se guarda?**
- **model_state_dict**: Todos los pesos y biases
- **optimizer_state_dict**: Estado del optimizador (por si queremos continuar entrenando)
- **Hiperparámetros**: Para reconstruir el modelo exactamente
- **Métricas**: Para referencia futura

#### 📈 Visualización de Métricas

```python
# Loss curves
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss')
plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label='Test Loss')

# Accuracy curves
plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, NUM_EPOCHS + 1), test_accuracies, label='Test Accuracy')
```

**Interpretación de las Gráficas**:

- **Loss decreciente**: El modelo está aprendiendo
- **Train accuracy > Test accuracy**: Normal, esperado
- **Gap muy grande**: Posible overfitting
- **Test loss aumentando**: Definitivamente overfitting

**Resultados de este Modelo**:
- Train Accuracy: 96.63%
- Test Accuracy: 68.00%
- Gap grande → overfitting presente

---

### 4️⃣ `model/training_metrics.json` - Métricas del Entrenamiento

**Propósito**: Almacena todas las métricas y resultados del entrenamiento.

#### 📊 Estructura del JSON

```json
{
    "training_info": {
        "epochs": 25,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "device": "cpu",
        "training_duration_seconds": 3230.94,
        "training_duration_formatted": "0:53:50"
    },
    "model_info": {
        "total_parameters": 1705732,
        "trainable_parameters": 1705732,
        "architecture": "MLP: 3072 -> 512 -> 256 -> 4"
    },
    "final_metrics": {
        "train_loss": 0.1057,
        "train_accuracy": 96.63,
        "test_loss": 2.185,
        "test_accuracy": 68.0
    }
}
```

**Análisis de Resultados**:

1. **Train Loss: 0.1057 (bajo)**
   - El modelo ha aprendido muy bien los datos de entrenamiento
   
2. **Test Loss: 2.185 (alto)**
   - En datos nuevos, el modelo tiene más error
   - Indicador claro de overfitting

3. **Train Accuracy: 96.63%**
   - Clasifica correctamente casi todas las imágenes de entrenamiento

4. **Test Accuracy: 68.0%**
   - En datos nuevos, solo acierta 68%
   - Gap del 28.63% indica memorización vs. generalización

**¿Cómo mejorar esto?**
- Regularización (Dropout, Weight Decay)
- Data Augmentation (rotaciones, flips)
- Modelo más simple (menos parámetros)
- Más datos de entrenamiento

---

### 5️⃣ `backend/main.py` - API FastAPI

**Propósito**: Servidor HTTP que expone el modelo para hacer predicciones.

#### 🚀 Inicialización de FastAPI

```python
app = FastAPI(title="CIFAR-10 MLP Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**¿Qué es CORS?**
- **Cross-Origin Resource Sharing**
- Permite que el frontend (puerto 3000) acceda al backend (puerto 8000)
- Sin CORS, el navegador bloquearía las peticiones

#### 🔧 Carga del Modelo

```python
model_path = '/app/model/cifar10_mlp.pth' if os.path.exists('/app/model/cifar10_mlp.pth') \
             else '../model/cifar10_mlp.pth'
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

**¿Por qué dos rutas?**
- `/app/model/cifar10_mlp.pth`: Ruta dentro de Docker
- `../model/cifar10_mlp.pth`: Ruta en desarrollo local
- `map_location=device`: Carga en CPU (para compatibilidad)

#### 🔍 Endpoint: `/predict`

```python
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leer imagen
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Transformar
    image_tensor = transform(image).unsqueeze(0)
    
    # Predecir
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return {
        "class_id": predicted.item(),
        "class_name": CLASS_NAMES[predicted.item()],
        "confidence": confidence.item(),
        "all_probabilities": probabilities[0].tolist()
    }
```

**Desglose Paso a Paso**:

1. **Recibir Imagen**:
   ```python
   file: UploadFile = File(...)  # FastAPI maneja multipart/form-data
   ```

2. **Convertir a PIL Image**:
   ```python
   image_bytes = await file.read()
   image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
   ```

3. **Aplicar Transformaciones**:
   ```python
   transform = transforms.Compose([
       transforms.Resize((32, 32)),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   image_tensor = transform(image).unsqueeze(0)
   ```
   - **Resize**: Asegura que la imagen sea 32x32
   - **unsqueeze(0)**: Añade dimensión de batch [1, 3, 32, 32]

4. **Inferencia**:
   ```python
   outputs = model(image_tensor)  # Logits [1, 4]
   probabilities = torch.nn.functional.softmax(outputs, dim=1)
   ```

**¿Qué es Softmax?**
```
Logits: [-2.3, 4.1, 0.5, -1.2]
         ↓ Softmax
Probabilities: [0.01, 0.92, 0.03, 0.04]  (suman 1.0)
```
- Convierte logits en probabilidades
- Fórmula: `softmax(x_i) = exp(x_i) / Σ exp(x_j)`

5. **Extraer Resultado**:
   ```python
   confidence, predicted = torch.max(probabilities, 1)
   ```
   - `predicted`: Índice de la clase con mayor probabilidad
   - `confidence`: Valor de esa probabilidad

#### 🌐 Endpoint: `/predict-external`

```python
@app.post("/predict-external")
async def predict_external(file: UploadFile = File(...)):
    original_size = image.size
    original_mode = image.mode
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((32, 32), Image.LANCZOS)
```

**Diferencias con `/predict`**:
- Acepta imágenes de **cualquier tamaño**
- Guarda información del procesamiento
- Convierte automáticamente a RGB (RGBA, grayscale, etc.)
- Usa LANCZOS para mejor calidad al redimensionar

**¿Por qué este endpoint adicional?**
- Permite usar imágenes de internet o cámara
- Más flexible para usuarios finales
- Transparente sobre el procesamiento realizado

---

### 6️⃣ `backend/requirements.txt` - Dependencias Backend

```
fastapi         # Framework web moderno
uvicorn         # Servidor ASGI para FastAPI
torch           # PyTorch para deep learning
torchvision     # Transformaciones de imágenes
python-multipart # Para manejar uploads de archivos
Pillow          # Procesamiento de imágenes
numpy           # Operaciones numéricas
```

**¿Por qué estas librerías?**
- **FastAPI + Uvicorn**: Servidor rápido y asíncrono
- **torch + torchvision**: Cargar modelo y preprocesar
- **Pillow**: Abrir y manipular imágenes
- **python-multipart**: Necesario para UploadFile

---

### 7️⃣ `backend/Dockerfile` - Containerización Backend

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código del backend
COPY main.py .

# Exponer puerto
EXPOSE 8000

# Comando para ejecutar
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Explicación Línea por Línea**:

1. **FROM python:3.10-slim**:
   - Imagen base de Python 3.10
   - `slim`: Versión ligera (menos librerías)
   - Reduce tamaño del contenedor

2. **WORKDIR /app**:
   - Establece directorio de trabajo
   - Todos los comandos siguientes se ejecutan aquí

3. **COPY requirements.txt .**:
   - Copia solo requirements primero
   - Aprovecha caché de Docker (eficiencia)

4. **RUN pip install --no-cache-dir -r requirements.txt**:
   - Instala dependencias Python
   - `--no-cache-dir`: No guarda caché, reduce tamaño

5. **COPY main.py .**:
   - Copia el código del servidor

6. **EXPOSE 8000**:
   - Documenta que el contenedor usa el puerto 8000
   - No abre el puerto (eso lo hace docker-compose)

7. **CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]**:
   - Comando por defecto al iniciar el contenedor
   - `--host 0.0.0.0`: Escucha en todas las interfaces
   - Necesario para acceder desde fuera del contenedor

---

### 8️⃣ `frontend/src/App.js` - Interfaz de Usuario React

**Propósito**: Aplicación web para cargar imágenes y visualizar predicciones.

#### 🎨 Estructura de Estados

```javascript
// Estados para el botón verde (dataset)
const [selectedFile, setSelectedFile] = useState(null);
const [preview, setPreview] = useState(null);
const [loading, setLoading] = useState(false);
const [result, setResult] = useState(null);
const [error, setError] = useState(null);

// Estados para el botón naranja (externas)
const [externalFile, setExternalFile] = useState(null);
const [externalPreview, setExternalPreview] = useState(null);
const [externalLoading, setExternalLoading] = useState(false);
const [externalResult, setExternalResult] = useState(null);
const [externalError, setExternalError] = useState(null);
```

**¿Qué son los estados en React?**
- Variables que cuando cambian, re-renderizan el componente
- `useState`: Hook para crear estado local
- Cada sección tiene sus propios estados (independientes)

#### 📤 Manejo de Selección de Archivo

```javascript
const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
        setSelectedFile(file);
        setPreview(URL.createObjectURL(file));  // Vista previa
        setResult(null);  // Limpiar resultado anterior
        setError(null);   // Limpiar error anterior
    }
};
```

**URL.createObjectURL(file)**:
- Crea una URL temporal que apunta al archivo local
- Permite mostrar la imagen sin subirla al servidor
- Se revoca automáticamente al cerrar la página

#### 🚀 Clasificación de Imagen

```javascript
const handleClassify = async () => {
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await axios.post('http://localhost:8000/predict', 
                                         formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        setResult(response.data);
    } catch (err) {
        setError('Error al clasificar la imagen: ' + err.message);
    } finally {
        setLoading(false);
    }
};
```

**Flujo de la Petición**:

1. **setLoading(true)**: Muestra "Clasificando..."
2. **FormData**: Formato para enviar archivos
3. **axios.post**: Petición HTTP POST al backend
4. **response.data**: JSON con la predicción
5. **setResult**: Actualiza UI con el resultado
6. **finally**: Se ejecuta siempre, success o error

#### 📊 Visualización de Top 3

```javascript
const getTop3 = () => {
    const probs = result.all_probabilities.map((prob, idx) => ({
        class_name: CLASS_NAMES[idx],
        probability: prob,
    }));
    
    return probs.sort((a, b) => b.probability - a.probability)
                .slice(0, 3);
};
```

**¿Qué hace?**
1. Combina probabilidades con nombres de clase
2. Ordena de mayor a menor probabilidad
3. Toma solo los 3 primeros
4. Retorna array: `[{class_name, probability}, ...]`

#### 🎨 Renderizado de Resultados

```javascript
{result && (
    <div style={styles.resultSection}>
        <h2>Resultado</h2>
        <p style={styles.className}>{result.class_name}</p>
        <p style={styles.confidence}>
            Confianza: {(result.confidence * 100).toFixed(2)}%
        </p>
        
        <h3>Top 3 Predicciones:</h3>
        {getTop3().map((item, idx) => (
            <div key={idx}>
                <span>{item.class_name}</span>
                <span>{(item.probability * 100).toFixed(2)}%</span>
            </div>
        ))}
    </div>
)}
```

**Renderizado Condicional**:
- `{result && ...}`: Solo renderiza si `result` existe
- Previene errores cuando no hay resultado aún

#### 🎨 Estilos CSS-in-JS

```javascript
const styles = {
    container: {
        maxWidth: '800px',
        margin: '50px auto',
        padding: '20px',
        fontFamily: 'Arial, sans-serif',
    },
    uploadButton: {
        backgroundColor: '#4CAF50',  // Verde
        color: 'white',
        padding: '12px 24px',
        borderRadius: '4px',
        cursor: 'pointer',
    },
    uploadButtonExternal: {
        backgroundColor: '#FF9800',  // Naranja
        // ... similar
    }
};
```

**¿Por qué CSS-in-JS?**
- Estilos viven con el componente
- No hay conflictos de nombres de clases
- Fácil de mantener para proyectos pequeños

---

### 9️⃣ `frontend/src/index.js` - Punto de Entrada

```javascript
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
```

**¿Qué hace este archivo?**
1. Importa React y ReactDOM
2. Crea un "root" React en el div con id="root"
3. Renderiza el componente `<App />` dentro del root
4. React toma control de ese div y gestiona el DOM

---

### 🔟 `frontend/public/index.html` - HTML Base

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>CIFAR-10 Classifier</title>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
```

**Simplicidad**:
- Solo un div: `<div id="root"></div>`
- React inyecta toda la app ahí
- No hay CSS, JS inline → todo gestionado por React

---

### 1️⃣1️⃣ `frontend/package.json` - Dependencias Frontend

```json
{
  "name": "cifar10-frontend",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  },
  "devDependencies": {
    "react-scripts": "5.0.1"
  }
}
```

**Dependencias Clave**:

- **react**: Librería principal
- **react-dom**: Integración con el DOM del navegador
- **axios**: Cliente HTTP para peticiones al backend
- **react-scripts**: Herramientas de desarrollo (webpack, babel, etc.)

**Scripts**:
- `npm start`: Inicia servidor de desarrollo (puerto 3000)
- `npm build`: Crea versión optimizada para producción

---

### 1️⃣2️⃣ `frontend/Dockerfile` - Containerización Frontend

```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package.json .
RUN npm install

COPY public ./public
COPY src ./src

EXPOSE 3000

CMD ["npm", "start"]
```

**Diferencias con Backend Dockerfile**:

1. **Imagen base**: `node:18-alpine` (Node.js en vez de Python)
2. **Gestión de dependencias**: `npm install` en vez de `pip`
3. **Puerto**: 3000 (estándar para React)
4. **Comando**: `npm start` (servidor de desarrollo)

**alpine**:
- Distribución Linux muy ligera
- Reduce tamaño de la imagen Docker
- ~5MB vs ~100MB para imágenes base completas

---

### 1️⃣3️⃣ `docker-compose.yml` - Orquestación de Contenedores

```yaml
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./model:/app/model:ro
    restart: always
    networks:
      - cifar10-network

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
    restart: always
    networks:
      - cifar10-network

networks:
  cifar10-network:
    driver: bridge
```

**Explicación Detallada**:

#### Backend Service

```yaml
backend:
  build: ./backend  # Construye Dockerfile en ./backend
  ports:
    - "8000:8000"  # Host:Container
  volumes:
    - ./model:/app/model:ro  # Montar carpeta model (read-only)
```

**Volumen ./model**:
- Monta la carpeta local `./model` dentro del contenedor en `/app/model`
- `:ro` = read-only (solo lectura)
- Permite al backend acceder al modelo sin copiarlo al contenedor
- Cambios en el modelo se reflejan inmediatamente

#### Frontend Service

```yaml
frontend:
  build: ./frontend
  ports:
    - "3000:3000"
  depends_on:
    - backend  # Espera que backend esté listo
```

**depends_on**:
- Asegura que backend inicie primero
- No espera que backend esté "listo", solo que inicie
- Para esperar que esté listo, se necesita healthcheck

#### Network

```yaml
networks:
  cifar10-network:
    driver: bridge
```

**¿Qué es una red bridge?**
- Red virtual privada entre contenedores
- Permite que contenedores se comuniquen por nombre
- Aislado del host (seguridad)

**Comunicación**:
- Frontend puede acceder a backend como `http://backend:8000`
- Pero desde el navegador se usa `http://localhost:8000` (fuera de Docker)

---

### 1️⃣4️⃣ `PLAN.md` - Plan de Implementación

**Propósito**: Documento de planificación y roadmap del proyecto.

#### Contenido Principal

1. **Objetivo del Proyecto**
2. **Arquitectura Propuesta**
3. **Pasos de Implementación**:
   - Fase 1: Preparación del Dataset
   - Fase 2: Modelo MLP
   - Fase 3: Backend FastAPI
   - Fase 4: Frontend React
   - Fase 5: Dockerización
   - Fase 6: Testing
4. **Detalles Técnicos** de cada componente
5. **Endpoints** de la API
6. **Comandos** para ejecutar

**Utilidad**:
- Sirve como referencia durante el desarrollo
- Documenta decisiones de diseño
- Facilita onboarding de nuevos desarrolladores

---

## 🔄 Flujo de Datos Completo

### Flujo de Predicción End-to-End

```
1. USUARIO
   │
   └─→ Selecciona imagen en navegador
       │
       └─→ [Frontend: App.js]
           │
           └─→ handleClassify() crea FormData
               │
               └─→ axios.post('http://localhost:8000/predict')
                   │
2. BACKEND
   │
   └─→ [Backend: main.py]
       │
       ├─→ Recibe imagen (FastAPI)
       ├─→ Convierte a PIL Image
       ├─→ Aplica transformaciones (resize, normalize)
       ├─→ Convierte a tensor [1, 3, 32, 32]
       │
       └─→ [Modelo: MLP]
           │
           ├─→ Forward pass
           │   └─→ fc1(3072 -> 512) + ReLU
           │       └─→ fc2(512 -> 256) + ReLU
           │           └─→ fc3(256 -> 4)
           │
           ├─→ Logits: [-2.3, 4.1, 0.5, -1.2]
           ├─→ Softmax: [0.01, 0.92, 0.03, 0.04]
           └─→ Predicción: clase 1 (automobile), 92% confianza
               │
3. RESPUESTA
   │
   └─→ JSON: {
           "class_id": 1,
           "class_name": "automobile",
           "confidence": 0.92,
           "all_probabilities": [0.01, 0.92, 0.03, 0.04]
       }
       │
4. FRONTEND
   │
   └─→ setResult(response.data)
       │
       └─→ Re-renderiza UI
           │
           ├─→ Muestra "AUTOMOBILE"
           ├─→ Muestra "Confianza: 92.00%"
           └─→ Muestra Top 3:
               1. automobile: 92.00%
               2. ship: 3.00%
               3. truck: 4.00%
```

---

## 💡 Conceptos Importantes Explicados

### 1. Red Neuronal Artificial (MLP)

**¿Qué es?**
- Sistema de procesamiento inspirado en el cerebro
- Compuesto por capas de neuronas artificiales
- Aprende patrones a partir de ejemplos

**Componentes**:
```
Neurona: f(Σ(w_i * x_i) + b)
         │   │    │     │
         │   │    │     └─ bias
         │   │    └─ input
         │   └─ peso
         └─ función de activación
```

**Proceso de Aprendizaje**:
1. Inicialización aleatoria de pesos
2. Forward pass: Calcular predicción
3. Calcular error (loss)
4. Backward pass: Calcular gradientes
5. Actualizar pesos para reducir error
6. Repetir hasta convergencia

### 2. Overfitting vs Underfitting

**Overfitting** (Sobreajuste):
- El modelo memoriza los datos de entrenamiento
- Funciona muy bien en train, mal en test
- **Síntomas**:
  - Train accuracy >> Test accuracy
  - Train loss << Test loss
- **Este proyecto**: 96.63% train vs 68% test

**Underfitting** (Subajuste):
- El modelo no aprende suficiente
- Funciona mal en train y test
- **Síntomas**:
  - Train accuracy baja
  - Test accuracy también baja

**Balance Ideal**:
- Train accuracy ≈ Test accuracy
- Generalización a datos nuevos

### 3. Batch Processing

**¿Por qué procesar en batches?**

**Sin Batches** (Stochastic Gradient Descent):
```
Para cada imagen:
    - Forward pass
    - Backward pass
    - Actualizar pesos
```
- Muy lento
- Actualización ruidosa

**Con Batches**:
```
Para cada batch de 64 imágenes:
    - Forward pass en paralelo
    - Calcular loss promedio
    - Backward pass
    - Actualizar pesos una vez
```
- Mucho más rápido (GPU parallelization)
- Actualizaciones más estables
- Uso eficiente de memoria

### 4. Learning Rate

**¿Qué controla?**
- Tamaño del paso al actualizar pesos
- `new_weight = old_weight - lr * gradient`

**Demasiado Alto** (lr > 0.01):
```
Loss
│     *
│    * *
│   *   *
│  *     *
└─────────── Iteraciones
No converge, oscila
```

**Demasiado Bajo** (lr < 0.0001):
```
Loss
│*
│ *
│  *
│   *___________
└─────────────── Iteraciones
Muy lento, puede estancarse
```

**Óptimo** (lr ≈ 0.001):
```
Loss
│*
│ **
│   ***
│      ****______
└─────────────── Iteraciones
Converge suavemente
```

### 5. Softmax y Probabilidades

**Función Softmax**:
```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Estabilidad numérica
    return exp_x / exp_x.sum()
```

**Ejemplo**:
```
Input (logits):  [-2.3,  4.1,  0.5, -1.2]
                    ↓ exp()
                 [0.10, 60.3, 1.65, 0.30]
                    ↓ normalize
Output (probs):  [0.002, 0.966, 0.026, 0.005]
                    ↓ sum = 1.0
```

**Propiedades**:
- Todas las probabilidades suman 1.0
- Amplifica diferencias (4.1 >> otros)
- Siempre positivas

### 6. Cross-Entropy Loss

**¿Qué mide?**
- Qué tan "lejos" está la predicción de la realidad

**Fórmula**:
```
Loss = -Σ y_true * log(y_pred)

Para clasificación:
Loss = -log(P(clase_correcta))
```

**Ejemplo**:
```
Imagen real: automobile (clase 1)
Predicciones: [0.01, 0.92, 0.03, 0.04]

Loss = -log(0.92) = 0.083
```

**Interpretación**:
- Predicción correcta con 99% → Loss ≈ 0.01
- Predicción correcta con 50% → Loss ≈ 0.69
- Predicción correcta con 10% → Loss ≈ 2.30

### 7. Docker y Containerización

**¿Qué es un contenedor?**
- Paquete autocontenido con aplicación + dependencias
- Aislado del sistema host
- Garantiza: "Funciona en mi máquina" = "Funciona en todas"

**Ventajas**:
- **Portabilidad**: Funciona igual en cualquier máquina
- **Aislamiento**: No interfiere con otras aplicaciones
- **Reproducibilidad**: Mismo ambiente siempre
- **Escalabilidad**: Fácil de replicar

**Docker vs VM**:
```
Virtual Machine:
[App] → [Guest OS] → [Hypervisor] → [Host OS] → [Hardware]
Pesado, lento

Docker Container:
[App] → [Docker Engine] → [Host OS] → [Hardware]
Ligero, rápido
```

### 8. API REST

**¿Qué es REST?**
- **RE**presentational **S**tate **T**ransfer
- Arquitectura para APIs web
- Usa verbos HTTP: GET, POST, PUT, DELETE

**Características**:
- **Stateless**: Cada petición es independiente
- **Client-Server**: Separación de responsabilidades
- **Cacheable**: Respuestas pueden ser cacheadas
- **Uniform Interface**: URLs y métodos estándar

**Ejemplo en este proyecto**:
```
POST /predict
Content-Type: multipart/form-data
Body: [imagen]

→ Backend procesa

Response:
{
  "class_name": "airplane",
  "confidence": 0.85
}
```

---

## 🚀 Instalación y Uso

### Opción 1: Con Docker (Recomendado)

```bash
# 1. Clonar o tener el proyecto
cd cifar10/

# 2. Preparar el dataset (solo primera vez)
python prepare_dataset.py

# 3. Entrenar el modelo (solo primera vez)
cd model/
python train.py
cd ..

# 4. Iniciar con Docker Compose
docker-compose up --build

# Acceder:
# - Frontend: http://localhost:3000
# - Backend: http://localhost:8000
```

### Opción 2: Sin Docker (Desarrollo Local)

**Backend**:
```bash
cd backend/
pip install -r requirements.txt
python main.py
```

**Frontend** (en otra terminal):
```bash
cd frontend/
npm install
npm start
```

### Comandos Útiles

```bash
# Ver logs de contenedores
docker-compose logs -f

# Detener contenedores
docker-compose down

# Reconstruir desde cero
docker-compose up --build --force-recreate

# Acceder a un contenedor
docker-compose exec backend bash
```

---

## 📊 Métricas y Rendimiento

### Resultados del Entrenamiento

| Métrica | Train | Test | Diferencia |
|---------|-------|------|------------|
| **Accuracy** | 96.63% | 68.00% | -28.63% |
| **Loss** | 0.106 | 2.185 | +2.079 |

### Interpretación

**Accuracy por Clase** (estimado):
- Airplane: ~70%
- Automobile: ~75%
- Ship: ~65%
- Truck: ~62%

**Problemas Detectados**:
1. **Overfitting severo**: Gap del 28% entre train y test
2. **Modelo simple**: MLP no es ideal para imágenes
3. **Sin regularización**: No hay Dropout ni Weight Decay

### Mejoras Posibles

1. **Arquitectura**:
   - Usar CNN en vez de MLP
   - CNNs son mejores para imágenes (capturan patrones espaciales)

2. **Regularización**:
   - Añadir Dropout (0.3-0.5)
   - Weight Decay en el optimizador
   - Early Stopping

3. **Data Augmentation**:
   - Rotaciones aleatorias
   - Flips horizontales
   - Cambios de brillo/contraste

4. **Más Datos**:
   - Usar las 10 clases completas
   - Aumentar dataset con augmentation

---

## 🔒 Seguridad y Consideraciones

### Seguridad

1. **CORS Abierto**:
   ```python
   allow_origins=["*"]  # Permite cualquier origen
   ```
   - **Problema**: Cualquier sitio web puede acceder
   - **Solución**: Restringir a dominios específicos en producción

2. **Sin Autenticación**:
   - Endpoints públicos sin protección
   - **Solución**: Añadir API keys o JWT

3. **Validación de Entrada**:
   - Solo valida tipo de archivo
   - **Solución**: Validar tamaño, formato, contenido

### Escalabilidad

**Limitaciones Actuales**:
- Un solo worker (uvicorn)
- Inferencia síncrona (bloquea mientras predice)
- Sin caché de resultados

**Mejoras**:
- Usar Gunicorn + múltiples workers
- Cola de tareas (Celery + Redis)
- Caché para imágenes ya procesadas

### Monitoreo

**Métricas a Trackear**:
- Latencia de predicciones
- Tasa de errores
- Uso de memoria/CPU
- Distribución de clases predichas

**Herramientas**:
- Prometheus + Grafana
- Sentry para errores
- Logs estructurados (JSON)

---

## 📖 Glosario de Términos

| Término | Definición |
|---------|-----------|
| **MLP** | Multi-Layer Perceptron, red neuronal fully-connected |
| **Epoch** | Pasar por todo el dataset de entrenamiento una vez |
| **Batch** | Subconjunto de datos procesados simultáneamente |
| **Learning Rate** | Tamaño del paso en la optimización |
| **Overfitting** | Modelo memoriza en vez de generalizar |
| **Forward Pass** | Cálculo de predicción (entrada → salida) |
| **Backward Pass** | Cálculo de gradientes (backpropagation) |
| **Logits** | Valores crudos antes de softmax |
| **Softmax** | Convierte logits en probabilidades |
| **Cross-Entropy** | Función de pérdida para clasificación |
| **Adam** | Algoritmo de optimización adaptativo |
| **ReLU** | Función de activación: max(0, x) |
| **Inference** | Usar el modelo para hacer predicciones |
| **Checkpoint** | Snapshot del modelo guardado |
| **Tensor** | Array multidimensional (PyTorch) |
| **Gradient** | Derivada del loss respecto a pesos |

---

## 🎓 Recursos Adicionales

### Documentación Oficial
- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Docker Documentation](https://docs.docker.com/)

### Tutoriales Recomendados
- [PyTorch 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Stanford CS231n - CNN for Visual Recognition](http://cs231n.stanford.edu/)

### Papers Relevantes
- [ImageNet Classification with Deep CNNs (AlexNet)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- [Adam Optimizer Paper](https://arxiv.org/abs/1412.6980)

---

## 📝 Conclusiones

Este proyecto implementa un clasificador de imágenes completo, desde la preparación de datos hasta el despliegue con Docker. Aunque el modelo MLP tiene limitaciones (overfitting, arquitectura simple), sirve como base excelente para:

1. **Aprender** los fundamentos de deep learning
2. **Entender** el ciclo completo de un proyecto ML
3. **Experimentar** con mejoras (CNN, regularización, etc.)
4. **Deployar** modelos con APIs modernas

El proyecto está bien estructurado, documentado y listo para ser extendido con mejoras más avanzadas.

---

**Última actualización**: Octubre 2025  
**Autor**: Proyecto CIFAR-10 MLP Classifier  
**Licencia**: Uso educativo
