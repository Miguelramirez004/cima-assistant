# CIMA Assistant

Sistema inteligente de consulta para formulación magistral y Centro de Información online de Medicamentos de la AEMPS (CIMA).

## Descripción

CIMA Assistant es una aplicación de Streamlit que proporciona dos funcionalidades principales:

1. **Formulación Magistral**: Genera formulaciones magistrales detalladas utilizando información de medicamentos registrados en CIMA.
2. **Consultas CIMA**: Permite interactuar con un chatbot especializado en información de medicamentos utilizando el modelo Sonar Pro de Perplexity AI.

La aplicación integra la API CIMA de la AEMPS (Agencia Española de Medicamentos y Productos Sanitarios) para formulaciones magistrales y utiliza el modelo Sonar Pro de Perplexity para proporcionar respuestas precisas y contextualizadas sobre medicamentos.

## Características

- **Formulación Magistral**:
  - Generación de formulaciones detalladas con estructura profesional
  - Búsqueda inteligente en la base de datos de CIMA
  - Referencias a fichas técnicas oficiales
  - Posibilidad de descargar las formulaciones como archivos Markdown

- **Consultas sobre medicamentos con Perplexity Sonar Pro**:
  - Chatbot conversacional con memoria de contexto
  - Respuestas avanzadas con capacidad de razonamiento
  - Información actualizada y precisa sobre medicamentos
  - Consultas sobre indicaciones, contraindicaciones, efectos adversos, etc.
  - Explicaciones detalladas y contextualizadas

## Instalación

1. Clone este repositorio:
   ```
   git clone https://github.com/Miguelramirez004/cima-assistant.git
   cd cima-assistant
   ```

2. Instale las dependencias:
   ```
   pip install -r requirements.txt
   ```

3. Configure las API keys:
   
   Cree un archivo `.env` en el directorio raíz y añada sus API keys:
   ```
   OPENAI_API_KEY=su_api_key_openai
   PERPLEXITY_API_KEY=su_api_key_perplexity
   ```

## Uso

Inicie la aplicación con Streamlit:
```
streamlit run app.py
```
o
```
streamlit run src/main.py
```

### Formulación Magistral

1. Seleccione la pestaña "Formulación Magistral"
2. Ingrese su consulta especificando:
   - Principio activo
   - Concentración deseada
   - Tipo de formulación 
3. Haga clic en "Consultar Formulación"
4. Revise la formulación generada y descárguela si lo desea

### Consultas CIMA con Perplexity

1. Seleccione la pestaña "Consultas CIMA"
2. Escriba su consulta sobre medicamentos en el campo de chat
3. El asistente proporcionará información detallada utilizando el modelo Sonar Pro
4. Mantenga conversaciones con contexto para consultas más complejas

## Tecnologías

- **Streamlit**: Framework para la interfaz de usuario
- **OpenAI API**: Modelos de lenguaje para generación de formulaciones
- **Perplexity Sonar Pro API**: Modelo avanzado de IA para consultas sobre medicamentos
- **CIMA API**: API oficial de la AEMPS para consulta de medicamentos
- **Python**: Lenguaje de programación principal
- **aiohttp/requests**: Clientes HTTP para comunicación con APIs

## Requisitos

- Python 3.8 o superior
- Conexión a Internet para acceder a las APIs
- API key de OpenAI y Perplexity

## Configuración en Streamlit Cloud

Para implementar en Streamlit Cloud, agregue las siguientes secrets:
- OPENAI_API_KEY
- PERPLEXITY_API_KEY

## Nota Legal

Esta aplicación proporciona información con fines educativos e informativos. No reemplaza la consulta médica profesional. Todas las formulaciones magistrales deben ser revisadas por un farmacéutico cualificado antes de su elaboración.

La información mostrada proviene de la API CIMA de la AEMPS y del modelo Sonar Pro de Perplexity. Esta aplicación no está afiliada oficialmente con la AEMPS ni con Perplexity AI.