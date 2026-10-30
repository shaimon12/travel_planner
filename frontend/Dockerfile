# ==============================================================================
# Stage 1: Build the React Application
# ==============================================================================
FROM node:20-alpine AS build

WORKDIR /app

# Copy package definitions and install dependencies
COPY package*.json ./
RUN npm install

# Copy application source code
COPY . .

# Build application bundle
RUN npm run build

# ==============================================================================
# Stage 2: Serve the Static Files with Nginx Alpine
# ==============================================================================
FROM nginx:alpine

# Copy custom Nginx configuration for React SPA routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy build artifacts from builder stage
COPY --from=build /app/dist /usr/share/nginx/html

# Expose HTTP port
EXPOSE 80

# Launch Nginx in foreground
CMD ["nginx", "-g", "daemon off;"]

