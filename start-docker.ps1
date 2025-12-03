# Script para ejecutar la aplicación con Docker en Windows

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "🚀 CIFAR-10 MLP Classifier - Docker Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar que best_model.pth existe
if (-Not (Test-Path "model\best_model.pth")) {
    Write-Host "❌ Error: model\best_model.pth no encontrado" -ForegroundColor Red
    Write-Host "Por favor, entrena el modelo primero ejecutando:" -ForegroundColor Yellow
    Write-Host "  cd model && python train.py" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Modelo encontrado: model\best_model.pth" -ForegroundColor Green
Write-Host ""

# Mostrar información del modelo
Write-Host "📊 Información del modelo:" -ForegroundColor Cyan
Get-Item model\best_model.pth | Format-Table Name, Length, LastWriteTime
Write-Host ""

# Detener contenedores existentes
Write-Host "🛑 Deteniendo contenedores existentes..." -ForegroundColor Yellow
docker-compose down
Write-Host ""

# Construir imágenes
Write-Host "🔨 Construyendo imágenes Docker..." -ForegroundColor Yellow
docker-compose build
Write-Host ""

# Iniciar contenedores
Write-Host "▶️  Iniciando contenedores..." -ForegroundColor Yellow
docker-compose up -d
Write-Host ""

# Esperar a que el backend esté listo
Write-Host "⏳ Esperando a que el backend esté listo..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Verificar que los contenedores están corriendo
Write-Host "📋 Estado de los contenedores:" -ForegroundColor Cyan
docker-compose ps
Write-Host ""

# Probar el health check del backend
Write-Host "🏥 Verificando health check del backend..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri http://localhost:8000/ -UseBasicParsing
    Write-Host "Backend respondiendo correctamente" -ForegroundColor Green
} catch {
    Write-Host "Backend aún iniciando..." -ForegroundColor Yellow
}
Write-Host ""

Write-Host "==========================================" -ForegroundColor Green
Write-Host "✅ Aplicación iniciada exitosamente!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "📱 Accede a la aplicación:" -ForegroundColor Cyan
Write-Host "   Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "   Backend API: http://localhost:8000" -ForegroundColor White
Write-Host "   API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "📝 Comandos útiles:" -ForegroundColor Cyan
Write-Host "   Ver logs:           docker-compose logs -f" -ForegroundColor White
Write-Host "   Ver logs backend:   docker-compose logs -f backend" -ForegroundColor White
Write-Host "   Ver logs frontend:  docker-compose logs -f frontend" -ForegroundColor White
Write-Host "   Detener todo:       docker-compose down" -ForegroundColor White
Write-Host "   Reiniciar:          docker-compose restart" -ForegroundColor White
Write-Host ""
