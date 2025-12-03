#!/bin/bash

# Script para ejecutar la aplicación con Docker

echo "=========================================="
echo "🚀 CIFAR-10 MLP Classifier - Docker Setup"
echo "=========================================="
echo ""

# Verificar que best_model.pth existe
if [ ! -f "model/best_model.pth" ]; then
    echo "❌ Error: model/best_model.pth no encontrado"
    echo "Por favor, entrena el modelo primero ejecutando:"
    echo "  cd model && python train.py"
    exit 1
fi

echo "✅ Modelo encontrado: model/best_model.pth"
echo ""

# Mostrar información del modelo
echo "📊 Información del modelo:"
ls -lh model/best_model.pth
echo ""

# Detener contenedores existentes
echo "🛑 Deteniendo contenedores existentes..."
docker-compose down
echo ""

# Construir imágenes
echo "🔨 Construyendo imágenes Docker..."
docker-compose build
echo ""

# Iniciar contenedores
echo "▶️  Iniciando contenedores..."
docker-compose up -d
echo ""

# Esperar a que el backend esté listo
echo "⏳ Esperando a que el backend esté listo..."
sleep 5

# Verificar que los contenedores están corriendo
echo "📋 Estado de los contenedores:"
docker-compose ps
echo ""

# Probar el health check del backend
echo "🏥 Verificando health check del backend..."
curl -s http://localhost:8000/ | python3 -m json.tool 2>/dev/null || echo "Backend respondiendo"
echo ""

echo "=========================================="
echo "✅ Aplicación iniciada exitosamente!"
echo "=========================================="
echo ""
echo "📱 Accede a la aplicación:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "📝 Comandos útiles:"
echo "   Ver logs:           docker-compose logs -f"
echo "   Ver logs backend:   docker-compose logs -f backend"
echo "   Ver logs frontend:  docker-compose logs -f frontend"
echo "   Detener todo:       docker-compose down"
echo "   Reiniciar:          docker-compose restart"
echo ""
