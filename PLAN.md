# Plan de Implementación: Clasificador CIFAR-10 con MLP

## Objetivo
Crear un clasificador de imágenes CIFAR-10 usando un MLP simple, con backend FastAPI, frontend React, todo dockerizado para uso local.

---

## Arquitectura del Sistema

```
cifar10/
├── data/                          # Datos del dataset
│   ├── train/                     # 19,200 imágenes (80%)
│   ├── test/                      # 4,800 imágenes (20%)
│   └── sample_images/             # 40 imágenes (10 por clase) para pruebas
├── model/
│   ├── mlp_model.py               # Definición del modelo MLP
│   ├── train.py                   # Script de entrenamiento
│   └── cifar10_mlp.pth            # Modelo entrenado (peso guardado)
├── backend/
│   ├── main.py                    # API FastAPI
│   ├── requirements.txt           # Dependencias Python
│   └── Dockerfile                 # Docker para backend
├── frontend/
│   ├── src/
│   │   ├── App.js                 # Componente principal React
│   │   └── index.js               # Entry point
│   ├── package.json               # Dependencias Node
│   ├── Dockerfile                 # Docker para frontend
│   └── public/
├── docker-compose.yml             # Orquestación de servicios
└── PLAN.md                        # Este archivo
```

---

## Pasos de Implementación

### Fase 1: Preparación del Dataset
**Archivo:** `prepare_dataset.py`

1. Descargar CIFAR-10 completo (50,000 imágenes de entrenamiento)
2. Filtrar solo 4 clases específicas:
   - **airplane** (clase 0)
   - **automobile** (clase 1)
   - **ship** (clase 8)
   - **truck** (clase 9)
3. Usar TODAS las instancias de estas 4 clases (~6,000 por clase = ~24,000 total)
4. Dividir en:
   - **Train:** 19,200 imágenes (80%)
   - **Test:** 4,800 imágenes (20%)
5. Copiar 10 imágenes de cada clase del test set a `data/sample_images/` (40 imágenes totales) para pruebas manuales
6. Guardar las imágenes organizadas por carpeta

**Clases seleccionadas:**
- 0: airplane
- 1: automobile
- 8: ship
- 9: truck

---

### Fase 2: Modelo MLP
**Archivos:** `model/mlp_model.py`, `model/train.py`

#### Arquitectura del MLP:
```
Input: 3072 (32x32x3 aplanado)
    ↓
Hidden Layer 1: 512 neuronas + ReLU
    ↓
Hidden Layer 2: 256 neuronas + ReLU (opcional)
    ↓
Output: 4 clases (Softmax)
```

#### Entrenamiento:
- **Optimizer:** Adam
- **Loss:** CrossEntropyLoss
- **Batch size:** 64
- **Epochs:** 20-30 (ajustable según tiempo)
- **Learning rate:** 0.001

#### Guardado:
- Guardar modelo entrenado como `cifar10_mlp.pth`
- Incluir métricas de accuracy en el log

---

### Fase 3: Backend FastAPI
**Archivos:** `backend/main.py`, `backend/requirements.txt`, `backend/Dockerfile`

#### Endpoints:

1. **GET /** 
   - Health check
   - Retorna: `{"status": "ok"}`

2. **POST /predict**
   - Recibe: imagen (multipart/form-data)
   - Procesa: redimensiona a 32x32, normaliza, aplana
   - Retorna: 
     ```json
     {
       "class_id": 0,
       "class_name": "airplane",
       "confidence": 0.85,
       "all_probabilities": [0.85, 0.10, 0.03, 0.02]
     }
     ```

#### Dependencias principales:
- fastapi
- uvicorn
- torch
- torchvision
- python-multipart
- Pillow

#### Docker:
- Base image: `python:3.10-slim`
- Puerto: 8000
- Volumen para cargar `cifar10_mlp.pth`

---

### Fase 4: Frontend React
**Archivos:** `frontend/src/App.js`, `frontend/Dockerfile`

#### Funcionalidad:
1. Interfaz simple con:
   - Botón para subir imagen
   - Preview de la imagen subida
   - Botón "Clasificar"
   - Mostrar resultado: clase predicha y confianza
   - Mostrar top 3 predicciones

2. Consumir API:
   - POST a `http://localhost:8000/predict`
   - Mostrar loading durante predicción
   - Manejo básico de errores

#### Dependencias:
- react
- axios (para llamadas HTTP)

#### Docker:
- Base image: `node:18-alpine`
- Puerto: 3000
- Servidor de desarrollo simple

---

### Fase 5: Docker Compose
**Archivo:** `docker-compose.yml`

#### Servicios:

1. **backend**
   - Build: `./backend`
   - Puerto: 8000:8000
   - Volúmenes: modelo y código
   - Restart: always

2. **frontend**
   - Build: `./frontend`
   - Puerto: 3000:3000
   - Depends on: backend
   - Restart: always

#### Red:
- Red interna para comunicación entre servicios

---

## Comandos de Uso

### 1. Preparar dataset:
```bash
python prepare_dataset.py
```

### 2. Entrenar modelo:
```bash
python model/train.py
```

### 3. Levantar aplicación:
```bash
docker-compose up --build
```

### 4. Acceder:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### 5. Detener:
```bash
docker-compose down
```

---

## Estimación de Tiempos

1. **Preparación dataset:** 5-10 minutos
2. **Entrenamiento modelo (CPU):** 20-40 minutos para 20-30 epochs (menos clases = más rápido)
3. **Desarrollo backend:** Ya incluido en scripts
4. **Desarrollo frontend:** Ya incluido en scripts
5. **Docker setup:** Build inicial ~10 minutos

**Tiempo total estimado:** ~45-90 minutos (principalmente entrenamiento)

---

## Notas Importantes

- **4 clases solamente:** Enfoque simplificado en vehículos/transporte (airplane, automobile, ship, truck)
- **Dataset completo por clase:** Usando todas las ~6,000 instancias disponibles por clase
- **Sin GPU:** El entrenamiento será en CPU (Ryzen 7), será lento pero funcional
- **Simplicidad:** No incluye validación avanzada, deployment profesional, ni optimizaciones
- **Local only:** Todo corre en localhost, sin HTTPS ni autenticación
- **Imágenes de prueba:** Las 40 imágenes copiadas (10 por clase) permiten probar el sistema sin buscar imágenes externas

---

## Próximos Pasos

1. ✅ Crear este plan
2. ⬜ Implementar `prepare_dataset.py`
3. ⬜ Implementar modelo y entrenamiento
4. ⬜ Crear backend FastAPI
5. ⬜ Crear frontend React
6. ⬜ Configurar Docker y docker-compose
7. ⬜ Probar sistema completo
