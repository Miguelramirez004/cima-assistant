# CIMA Assistant Setup Guide

Este documento proporciona instrucciones detalladas para configurar y ejecutar CIMA Assistant correctamente.

## Requisitos Previos

Para ejecutar CIMA Assistant necesita:

- Python 3.9 o superior
- Una clave API de OpenAI (única clave necesaria; se usa GPT-4o mini)
- Conexión a Internet para acceder a la API de CIMA (AEMPS) y a OpenAI

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

### 4. Configurar la clave API

Existen tres formas de proporcionar su clave API:

#### Opción 1: Archivo .env
Cree un archivo `.env` en el directorio raíz con el siguiente contenido:

```
OPENAI_API_KEY=su_clave_openai_aquí
```

#### Opción 2: Variables de entorno
Configure la variable de entorno directamente:

```bash
# En Windows (PowerShell):
$env:OPENAI_API_KEY="su_clave_openai_aquí"

# En Windows (CMD):
set OPENAI_API_KEY=su_clave_openai_aquí

# En macOS/Linux:
export OPENAI_API_KEY=su_clave_openai_aquí
```

#### Opción 3: Para despliegue en Streamlit Cloud
En la configuración de secretos de Streamlit Cloud, añada:

```
OPENAI_API_KEY=su_clave_openai_aquí
```

### 5. Obtener la clave API

#### OpenAI API
1. Visite [OpenAI API](https://platform.openai.com/account/api-keys)
2. Cree una cuenta o inicie sesión
3. Vaya a "API keys" y cree una nueva clave API

## Ejecutar la Aplicación

### Modo Local

Para ejecutar la aplicación en modo local:

```bash
streamlit run app.py
```

La aplicación estará disponible en `http://localhost:8501` por defecto.

## Funcionalidades y Modos

### Modos de búsqueda para formulación
La aplicación ofrece dos modos de búsqueda para la sección de formulación magistral:

1. **Búsqueda estándar**: Utiliza el método tradicional de búsqueda en la API CIMA
2. **Búsqueda avanzada**: Resuelve el principio activo contra el catálogo oficial
   de la AEMPS (`maestras`) y recupera por identificador exacto (`idpractiv1`)

Puede alternar entre estos modos en la barra lateral de la aplicación.

### Consultas CIMA (RAG sobre la API oficial)
La sección de consultas CIMA ejecuta un grafo de recuperación sobre la API REST
oficial de CIMA:

1. Clasifica la intención de la consulta y la mapea a la sección correspondiente
   de la ficha técnica (4.1 indicaciones, 4.2 posología, 4.3 contraindicaciones,
   4.5 interacciones, 4.8 efectos adversos, etc.)
2. Resuelve el principio activo contra el catálogo oficial (`maestras`)
3. Recupera medicamentos por id exacto o mediante búsqueda full-text por sección
   (`buscarEnFichaTecnica`)
4. Descarga las secciones relevantes en texto plano (`docSegmentado/contenido`)
5. Redacta la respuesta con GPT-4o mini citando las fichas técnicas utilizadas

La traza de recuperación puede visualizarse activando "Mostrar proceso de
razonamiento" en la barra lateral.

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

1. Verifique que la clave API de OpenAI esté correctamente configurada
2. Compruebe que la clave no ha caducado o alcanzado límites de uso
3. Compruebe que https://cima.aemps.es/ está accesible desde su red

## Notas de Implementación

La aplicación utiliza varias estrategias para garantizar un funcionamiento robusto:

1. **Gestión mejorada de ciclos de eventos**: Cada operación asíncrona se ejecuta en un ciclo de eventos aislado para evitar conflictos.
2. **Límites de conexión**: Se han implementado límites de conexión para evitar saturar las APIs.
3. **Reintentos inteligentes**: Las solicitudes HTTP implementan estrategias de reintento con backoff exponencial.
4. **Limpieza de recursos**: Garantizamos la liberación adecuada de recursos para evitar fugas de memoria.
5. **Saneamiento de contenido**: Todo el contenido recuperado y generado pasa por
   las utilidades de `security.py` (anti-XSS, anti-inyección de prompts, límites
   de tamaño).

## Contacto y Soporte

Si tiene problemas o preguntas, por favor abra un issue en el repositorio de GitHub.
