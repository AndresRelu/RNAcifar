# 🐳 GUÍA DE EJECUCIÓN CON DOCKER

## 📋 Requisitos Previos

1. ✅ **Docker Desktop** instalado y en ejecución
2. ✅ **Docker Compose** instalado (viene con Docker Desktop)
3. ✅ Modelo entrenado (`model/best_model.pth` debe existir)

## 🚀 Inicio Rápido

### Opción 1: Script Automático (Recomendado)

#### En WSL/Linux:
```bash
chmod +x start-docker.sh
./start-docker.sh
```

#### En Windows PowerShell:
```powershell
.\start-docker.ps1
```

#### En Windows CMD:
```cmd
docker-compose up -d --build
```

### Opción 2: Comandos Manuales

```bash
# 1. Construir las imágenes
docker-compose build

# 2. Iniciar los contenedores
docker-compose up -d

# 3. Ver los logs
docker-compose logs -f
```

---

## 🌐 Acceder a la Aplicación

Una vez iniciado, accede a:

- **Frontend (Interfaz Web):** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Documentación (Swagger):** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/

---

## 📊 Verificar Estado

```bash
# Ver contenedores en ejecución
docker-compose ps

# Ver logs en tiempo real
docker-compose logs -f

# Ver logs solo del backend
docker-compose logs -f backend

# Ver logs solo del frontend
docker-compose logs -f frontend
```

---

## 🛠️ Comandos Útiles

### Detener la aplicación:
```bash
docker-compose down
```

### Reiniciar la aplicación:
```bash
docker-compose restart
```

### Reconstruir y reiniciar (después de cambios en código):
```bash
docker-compose down
docker-compose up -d --build
```

### Detener y eliminar todo (incluyendo volúmenes):
```bash
docker-compose down -v
```

### Acceder al contenedor del backend:
```bash
docker-compose exec backend bash
```

### Acceder al contenedor del frontend:
```bash
docker-compose exec frontend sh
```

---

## 🔍 Probar el Backend

### Desde la terminal:

#### Test de salud:
```bash
curl http://localhost:8000/
```

#### Predecir una imagen:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@ruta/a/tu/imagen.jpg"
```

---

## 📁 Estructura de Docker

```
cifar10/
├── docker-compose.yml          # Orquestación de contenedores
├── backend/
│   ├── Dockerfile              # Imagen del backend
│   ├── main.py                 # API FastAPI
│   └── requirements.txt        # Dependencias Python
├── frontend/
│   ├── Dockerfile              # Imagen del frontend
│   ├── package.json            # Dependencias Node.js
│   └── src/                    # Código React
└── model/
    └── best_model.pth          # Mejor modelo entrenado ⭐
```

---

## ⚙️ Configuración de Docker Compose

El `docker-compose.yml` configura:

### Backend:
- **Puerto:** 8000
- **Volumen:** `./model:/app/model:ro` (solo lectura)
- **Red:** cifar10-network

### Frontend:
- **Puerto:** 3000
- **Depende de:** backend
- **Red:** cifar10-network

---

## 🐛 Solución de Problemas

### 1. Error "Puerto ya en uso"

```bash
# Detener contenedores que usen el puerto
docker-compose down

# O cambiar el puerto en docker-compose.yml
ports:
  - "8001:8000"  # Cambiar 8000 por otro puerto
```

### 2. Error "Modelo no encontrado"

Verifica que `model/best_model.pth` exista:
```bash
ls -lh model/best_model.pth
```

Si no existe, entrena el modelo:
```bash
cd model
python train.py
cd ..
```

### 3. Error "Cannot connect to Docker daemon"

Asegúrate de que Docker Desktop está en ejecución.

### 4. Contenedor se detiene inmediatamente

Ver los logs para identificar el error:
```bash
docker-compose logs backend
docker-compose logs frontend
```

### 5. Frontend no puede conectar con Backend

Verifica que ambos contenedores están en la misma red:
```bash
docker network ls
docker network inspect cifar10_cifar10-network
```

---

## 🔄 Actualizar el Modelo

Si entrenas un nuevo modelo mejor:

1. Entrena el modelo:
```bash
cd model
python train.py
cd ..
```

2. Reinicia solo el backend (no necesita rebuild):
```bash
docker-compose restart backend
```

3. Verifica que cargó el nuevo modelo:
```bash
docker-compose logs backend | grep "Mejor modelo"
```

---

## 📈 Información del Modelo Actual

El modelo cargado mostrará en los logs:

```
✓ Mejor modelo cargado exitosamente
  - Epoch: [número]
  - Test Accuracy: [porcentaje]%
  - Test Loss: [valor]
```

Para verlo:
```bash
docker-compose logs backend | grep -A 3 "Mejor modelo"
```

---

## 🎯 Flujo Completo de Uso

1. **Entrenar modelo** (si no existe):
   ```bash
   cd model && python train.py && cd ..
   ```

2. **Iniciar Docker**:
   ```bash
   docker-compose up -d --build
   ```

3. **Abrir navegador**:
   - Frontend: http://localhost:3000

4. **Subir imagen** y obtener predicción

5. **Ver logs** (opcional):
   ```bash
   docker-compose logs -f
   ```

6. **Detener** cuando termines:
   ```bash
   docker-compose down
   ```

---

## 💡 Tips

- El modelo se monta como **volumen de solo lectura** (`ro`), no se modificará dentro del contenedor
- Los logs se ven en tiempo real con `-f` (follow)
- Usa `docker-compose restart` para reiniciar sin reconstruir (más rápido)
- Usa `--build` solo cuando cambies código o dependencias

---

## 🎉 ¡Todo Listo!

Tu aplicación de clasificación CIFAR-10 con el **mejor modelo entrenado** está lista para usar con Docker.

El modelo mejorado con:
- ✅ 3 capas ocultas (512-256-128)
- ✅ Batch Normalization
- ✅ Dropout 0.4
- ✅ ~70-75% test accuracy

¡Disfruta clasificando imágenes! 🚀
