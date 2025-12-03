FROM node:18-alpine

WORKDIR /app

# Copiar package.json
COPY package.json .

# Instalar dependencias
RUN npm install

# Copiar código
COPY public ./public
COPY src ./src

# Exponer puerto
EXPOSE 3000

# Comando para ejecutar
CMD ["npm", "start"]
