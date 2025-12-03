# Configuración para AWS

## Problema resuelto
El "Network Error" ocurría porque el frontend intentaba conectarse a `http://localhost:8000`, pero en AWS el backend no está en localhost.

## Solución implementada

### 1. Variables de entorno configuradas
- El frontend ahora usa `REACT_APP_API_URL` para conectarse al backend
- Por defecto usa `http://localhost:8000` para desarrollo local
- En AWS debe apuntar a la IP pública o dominio de tu instancia EC2

### 2. Para desplegar en AWS

**IMPORTANTE**: Usa tu IP pública de AWS (sin los símbolos < >)

```bash
# En tu instancia EC2, exporta la variable con tu IP pública
# Ejemplo: export REACT_APP_API_URL=http://13.58.211.235:8000
export REACT_APP_API_URL=http://TU_IP_PUBLICA_AWS:8000

# Luego reconstruye y despliega (IMPORTANTE: usar --no-cache para forzar rebuild)
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

**Nota**: NO uses `http://backend:8000` porque eso solo funciona dentro de Docker. El navegador necesita la IP pública de tu servidor AWS.

### 3. Verificar configuración de seguridad en AWS
Asegúrate de que el Security Group permita:
- Puerto 80 (HTTP) para el frontend
- Puerto 8000 para el backend
- Desde cualquier IP (0.0.0.0/0) si quieres acceso público

### 4. CORS configurado
El backend ya tiene CORS configurado para aceptar peticiones de cualquier origen (`allow_origins=["*"]`), así que no deberías tener problemas de CORS.

## Para desarrollo local
No necesitas hacer nada, funcionará con localhost por defecto.

## Para verificar
1. En AWS, abre la consola del navegador (F12)
2. Ve a la pestaña Network
3. Intenta clasificar una imagen
4. Verifica que la petición vaya a `http://<TU_IP_AWS>:8000/predict` y no a localhost
