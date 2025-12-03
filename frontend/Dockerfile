FROM node:18-alpine

WORKDIR /app

# Copiar package.json
COPY package.json .

# Instalar dependencias
RUN npm install

# Copiar código
COPY public ./public
COPY src ./src

# Comando para ejecutar
CMD ["npm", "start"]
