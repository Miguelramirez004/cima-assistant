# CIMA Assistant

Sistema inteligente de consulta para formulación magistral y Centro de Información online de Medicamentos de la AEMPS (CIMA).

## Descripción

CIMA Assistant es una aplicación de Streamlit que proporciona dos funcionalidades principales:

1. **Formulación Magistral**: Genera formulaciones magistrales detalladas utilizando información de medicamentos registrados en CIMA.
2. **Consultas CIMA**: Permite interactuar con un chatbot especializado en información de medicamentos.

La aplicación integra la API CIMA de la AEMPS (Agencia Española de Medicamentos y Productos Sanitarios) y utiliza modelos de OpenAI para proporcionar respuestas precisas y contextualizadas.

## Características

- **Formulación Magistral**:
  - Generación de formulaciones detalladas con estructura profesional
  - Búsqueda inteligente en la base de datos de CIMA
  - Referencias a fichas técnicas oficiales
  - Posibilidad de descargar las formulaciones como archivos Markdown

- **Consultas sobre medicamentos**:
  - Chatbot conversacional con memoria de contexto
  - Consultas sobre indicaciones, contraindicaciones, efectos adversos, etc.
  - Referencias a fuentes oficiales de CIMA
  - Enlaces a fichas técnicas

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

3. Configure la API key de OpenAI:
   
   Cree un archivo `.env` en el directorio raíz y añada su API key de OpenAI:
   ```
   OPENAI_API_KEY=su_api_key
   ```

## Uso

Inicie la aplicación con Streamlit:
```
streamlit run app.py
```

### Formulación Magistral

1. Seleccione la pestaña "Formulación Magistral"
2. Ingrese su consulta especificando:
   - Principio activo
   - Concentración deseada
   - Tipo de formulación 
3. Haga clic en "Consultar Formulación"
4. Revise la formulación generada y descárguela si lo desea

### Consultas CIMA

1. Seleccione la pestaña "Consultas CIMA"
2. Escriba su consulta sobre medicamentos en el campo de chat
3. El asistente proporcionará información basada en datos oficiales de CIMA

## Tecnologías

- **Streamlit**: Framework para la interfaz de usuario
- **OpenAI API**: Modelos de lenguaje para generación de respuestas
- **CIMA API**: API oficial de la AEMPS para consulta de medicamentos
- **Python**: Lenguaje de programación principal
- **aiohttp**: Cliente HTTP asíncrono para comunicación con APIs

## Requisitos

- Python 3.8 o superior
- Conexión a Internet para acceder a las APIs
- API key de OpenAI

## Nota Legal

Esta aplicación proporciona información con fines educativos e informativos. No reemplaza la consulta médica profesional. Todas las formulaciones magistrales deben ser revisadas por un farmacéutico cualificado antes de su elaboración.

La información mostrada proviene de la API CIMA de la AEMPS, pero esta aplicación no está afiliada oficialmente con la AEMPS.