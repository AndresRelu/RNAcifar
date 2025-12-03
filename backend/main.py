import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from torchvision import transforms

# Definir el modelo MLP mejorado
class MLP(nn.Module):
    def __init__(self, input_size=3072, hidden_sizes=[512, 256, 128], num_classes=4, dropout_rate=0.4):
        super(MLP, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
        # Primera capa
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Segunda capa
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Tercera capa
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Capa de salida
        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)
        
    def forward(self, x):
        # Aplanar la imagen de (batch, 3, 32, 32) a (batch, 3072)
        x = x.view(x.size(0), -1)
        
        # Primera capa oculta
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Segunda capa oculta
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Tercera capa oculta
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        # Capa de salida
        x = self.fc4(x)
        
        return x

# Inicializar FastAPI
app = FastAPI(title="CIFAR-10 MLP Classifier")

# CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clases del modelo
CLASS_NAMES = ['airplane', 'automobile', 'ship', 'truck']

# Cargar el modelo
device = torch.device('cpu')
model = MLP()  # Usa los parámetros por defecto que coinciden con el modelo entrenado

# Ruta del modelo (funciona tanto local como en Docker) - USA BEST_MODEL
import os
# En Docker, el volumen monta ./model en /app/model
model_path = '/app/model/best_model.pth'

# Verificar si existe el archivo
if not os.path.exists(model_path):
    print(f"✗ Archivo de modelo no encontrado en: {model_path}")
    print(f"✗ Contenido de /app/model/:")
    if os.path.exists('/app/model'):
        print(os.listdir('/app/model'))
    raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

try:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Mostrar información del modelo cargado
    print("✓ Mejor modelo cargado exitosamente")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Test Accuracy: {checkpoint.get('test_accuracy', 'N/A'):.2f}%")
    print(f"  - Test Loss: {checkpoint.get('test_loss', 'N/A'):.4f}")
except Exception as e:
    print(f"✗ Error cargando modelo: {e}")
    raise

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predice la clase de una imagen
    
    Args:
        file: Imagen a clasificar
        
    Returns:
        JSON con class_id, class_name, confidence y all_probabilities
    """
    try:
        # Leer y procesar la imagen
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Aplicar transformaciones
        image_tensor = transform(image).unsqueeze(0)
        
        # Hacer predicción
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        # Preparar respuesta
        class_id = predicted.item()
        class_name = CLASS_NAMES[class_id]
        confidence_value = confidence.item()
        all_probs = probabilities[0].tolist()
        
        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence_value, 4),
            "all_probabilities": [round(p, 4) for p in all_probs]
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict-external")
async def predict_external(file: UploadFile = File(...)):
    """
    Predice la clase de una imagen externa de cualquier tamaño.
    La imagen será redimensionada automáticamente a 32x32 con 3 canales RGB.
    
    Args:
        file: Imagen a clasificar (cualquier tamaño)
        
    Returns:
        JSON con class_id, class_name, confidence, all_probabilities y processing_info
    """
    try:
        # Leer la imagen
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Obtener dimensiones originales
        original_size = image.size
        original_mode = image.mode
        
        # Convertir a RGB si no lo está (maneja RGBA, L, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionar a 32x32
        image = image.resize((32, 32), Image.LANCZOS)
        
        # Aplicar transformaciones (normalización)
        image_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])(image).unsqueeze(0)
        
        # Hacer predicción
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        # Preparar respuesta
        class_id = predicted.item()
        class_name = CLASS_NAMES[class_id]
        confidence_value = confidence.item()
        all_probs = probabilities[0].tolist()
        
        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence_value, 4),
            "all_probabilities": [round(p, 4) for p in all_probs],
            "processing_info": {
                "original_size": f"{original_size[0]}x{original_size[1]}",
                "original_mode": original_mode,
                "processed_size": "32x32",
                "processed_mode": "RGB"
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
