FROM node:18-alpine

WORKDIR /app

# Argumento de build para la URL del API
ARG REACT_APP_API_URL=http://localhost:8000

# Copiar package.json
COPY package.json .

# Instalar dependencias
RUN npm install

# Copiar código
COPY public ./public
COPY src ./src

# Establecer la variable de entorno
ENV REACT_APP_API_URL=$REACT_APP_API_URL

# Comando para ejecutar
CMD ["npm", "start"]
