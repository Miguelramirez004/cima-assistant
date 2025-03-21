# CIMA Assistant Setup Guide

Este documento proporciona instrucciones detalladas para configurar y ejecutar CIMA Assistant correctamente.

## Requisitos Previos

Para ejecutar CIMA Assistant necesita:

- Python 3.9 o superior
- Una clave API de OpenAI (para formulaciones magistrales)
- Una clave API de Perplexity (para consultas CIMA)
- Conexión a Internet para acceder a las APIs

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

### 4. Configurar las claves API

Existen tres formas de proporcionar sus claves API:

#### Opción 1: Archivo .env
Cree un archivo `.env` en el directorio raíz con el siguiente contenido:

```
OPENAI_API_KEY=su_clave_openai_aquí
PERPLEXITY_API_KEY=su_clave_perplexity_aquí
```

#### Opción 2: Variables de entorno
Configure las variables de entorno directamente:

```bash
# En Windows (PowerShell):
$env:OPENAI_API_KEY="su_clave_openai_aquí"
$env:PERPLEXITY_API_KEY="su_clave_perplexity_aquí"

# En Windows (CMD):
set OPENAI_API_KEY=su_clave_openai_aquí
set PERPLEXITY_API_KEY=su_clave_perplexity_aquí

# En macOS/Linux:
export OPENAI_API_KEY=su_clave_openai_aquí
export PERPLEXITY_API_KEY=su_clave_perplexity_aquí
```

#### Opción 3: Para despliegue en Streamlit Cloud
En la configuración de secretos de Streamlit Cloud, añada:

```
OPENAI_API_KEY=su_clave_openai_aquí
PERPLEXITY_API_KEY=su_clave_perplexity_aquí
```

### 5. Obtener claves API

#### OpenAI API
1. Visite [OpenAI API](https://platform.openai.com/account/api-keys)
2. Cree una cuenta o inicie sesión
3. Vaya a "API keys" y cree una nueva clave API

#### Perplexity API
1. Visite [Perplexity AI](https://www.perplexity.ai/)
2. Cree una cuenta o inicie sesión
3. Vaya a configuración y busque la sección de API para obtener su clave

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

## Funcionalidades y Modos

### Modos de búsqueda para formulación
La aplicación ofrece dos modos de búsqueda para la sección de formulación magistral:

1. **Búsqueda estándar**: Utiliza el método tradicional de búsqueda en la API CIMA
2. **Búsqueda avanzada**: Implementa un sistema mejorado basado en técnicas avanzadas de relevancia

Puede alternar entre estos modos en la barra lateral de la aplicación.

### Consultas CIMA con Perplexity
La sección de consultas CIMA utiliza ahora el modelo Sonar Pro de Perplexity AI, que proporciona:

- Respuestas más precisas y contextualizadas
- Información actualizada sobre medicamentos
- Capacidad mejorada de razonamiento para consultas complejas
- Manejo de conversaciones con contexto

## Solución de Problemas Comunes

### Error: Event loop is closed

Si encuentra errores relacionados con "Event loop is closed", asegúrese de estar utilizando las versiones más recientes de la aplicación, que incluyen mejoras en la gestión de ciclos de eventos asíncronos. Si persiste, reinicie la aplicación.

### Error al recuperar información del prospecto

Las últimas actualizaciones mejoran significativamente la recuperación de información de prospectos de la base de datos CIMA. Si encuentra algún problema:

1. Verifique su conexión a Internet
2. Asegúrese de que está utilizando un término de búsqueda específico
3. Intente reiniciar la aplicación

### Errores de API

Si encuentra errores relacionados con las APIs:

1. Verifique que las claves API estén correctamente configuradas
2. Compruebe que las claves no han caducado o alcanzado límites de uso
3. Si usa Perplexity, verifique que su cuenta tenga acceso al modelo Sonar Pro

## Notas de Implementación

La aplicación utiliza varias estrategias para garantizar un funcionamiento robusto:

1. **Gestión mejorada de ciclos de eventos**: Cada operación asíncrona se ejecuta en un ciclo de eventos aislado para evitar conflictos.
2. **Límites de conexión**: Se han implementado límites de conexión para evitar saturar las APIs.
3. **Reintentos inteligentes**: Las solicitudes HTTP implementan estrategias de reintento con backoff exponencial.
4. **Limpieza de recursos**: Garantizamos la liberación adecuada de recursos para evitar fugas de memoria.

## Contacto y Soporte

Si tiene problemas o preguntas, por favor abra un issue en el repositorio de GitHub.