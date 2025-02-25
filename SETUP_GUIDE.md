# CIMA Assistant Setup Guide

Este documento proporciona instrucciones detalladas para configurar y ejecutar CIMA Assistant correctamente.

## Requisitos Previos

Para ejecutar CIMA Assistant necesita:

- Python 3.9 o superior
- Una clave API de OpenAI
- Conexión a Internet para acceder a la API CIMA

## Instrucciones de Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/Miguelramirez004/cima-assistant.git
cd cima-assistant
```

### 2. Crear un entorno virtual

Recomendamos utilizar un entorno virtual para aislar las dependencias:

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar la clave API de OpenAI

Existen tres formas de proporcionar su clave API de OpenAI:

#### Opción 1: Archivo .env
Cree un archivo `.env` en el directorio raíz con el siguiente contenido:

```
OPENAI_API_KEY=su_clave_api_aquí
```

#### Opción 2: Variable de entorno
Configure la variable de entorno directamente:

```bash
# En Windows (PowerShell):
$env:OPENAI_API_KEY="su_clave_api_aquí"

# En Windows (CMD):
set OPENAI_API_KEY=su_clave_api_aquí

# En macOS/Linux:
export OPENAI_API_KEY=su_clave_api_aquí
```

#### Opción 3: Para despliegue en Streamlit Cloud
En la configuración de secretos de Streamlit Cloud, añada:

```
OPENAI_API_KEY=su_clave_api_aquí
```

## Ejecutar la Aplicación

### Modo Local

Para ejecutar la aplicación en modo local:

```bash
streamlit run app.py
```

O alternativamente:

```bash
streamlit run src/main.py
```

La aplicación estará disponible en `http://localhost:8501` por defecto.

## Solución de Problemas Comunes

### Error: Event loop is closed

Si encuentra errores relacionados con "Event loop is closed", asegúrese de estar utilizando las versiones más recientes de la aplicación, que incluyen mejoras en la gestión de ciclos de eventos asíncronos. Si persiste, reinicie la aplicación.

### Error al recuperar información del prospecto

Las últimas actualizaciones mejoran significativamente la recuperación de información de prospectos de la base de datos CIMA. Si encuentra algún problema:

1. Verifique su conexión a Internet
2. Asegúrese de que está utilizando un término de búsqueda específico
3. Intente reiniciar la aplicación

### Errores SSL/Transport

Si encuentra errores relacionados con transporte SSL, las mejoras en la gestión de conexiones deberían resolverlos. En caso contrario:

1. Actualice las dependencias: `pip install -r requirements.txt --upgrade`
2. Verifique que no hay restricciones de firewall/proxy en su red

## Notas de Implementación

La aplicación utiliza varias estrategias para garantizar un funcionamiento robusto:

1. **Gestión mejorada de ciclos de eventos**: Cada operación asíncrona se ejecuta en un ciclo de eventos aislado para evitar conflictos.
2. **Límites de conexión**: Se han implementado límites de conexión para evitar saturar la API CIMA.
3. **Reintentos inteligentes**: Las solicitudes HTTP implementan estrategias de reintento con backoff exponencial.
4. **Limpieza de recursos**: Garantizamos la liberación adecuada de recursos para evitar fugas de memoria.

## Contacto y Soporte

Si tiene problemas o preguntas, por favor abra un issue en el repositorio de GitHub.