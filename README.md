# CIMA Assistant

Sistema inteligente de consulta para formulación magistral y Centro de Información online de Medicamentos de la AEMPS (CIMA).

## Descripción

CIMA Assistant es una aplicación de Streamlit que proporciona tres funcionalidades principales:

1. **Formulación Magistral**: Genera formulaciones magistrales detalladas utilizando información de medicamentos registrados en CIMA.
2. **Consultas CIMA**: Chatbot RAG que responde sobre medicamentos usando exclusivamente la API REST oficial de CIMA (búsqueda por principio activo y búsqueda full-text por sección de ficha técnica) y GPT-4o mini para la redacción.
3. **Prospectos**: Genera prospectos en el formato oficial de la AEMPS a partir del prospecto real registrado en CIMA.

Toda la información clínica procede de la API CIMA de la AEMPS (Agencia Española de Medicamentos y Productos Sanitarios); el modelo de lenguaje solo redacta a partir de ese contexto recuperado y cita las fichas técnicas utilizadas.

## Características

- **Formulación Magistral**:
  - Generación de formulaciones detalladas con estructura profesional
  - Búsqueda inteligente en la base de datos de CIMA, con resolución del principio
    activo contra el catálogo oficial (`maestras`) y recuperación exacta por
    `idpractiv1`
  - Referencias a fichas técnicas oficiales
  - Posibilidad de descargar las formulaciones como archivos Markdown

- **Consultas CIMA (RAG sobre la API oficial)**:
  - Grafo de recuperación tipado con Pydantic: análisis de intención →
    resolución de principio activo → recuperación → lectura de secciones →
    generación
  - Detección de la sección relevante de la ficha técnica (4.1 indicaciones,
    4.2 posología, 4.3 contraindicaciones, 4.5 interacciones, 4.8 efectos
    adversos, etc.)
  - Búsqueda inversa por contenido con `buscarEnFichaTecnica` (p. ej. «¿qué
    medicamentos están indicados para la hipertensión?»)
  - Visualización del proceso de recuperación (traza del grafo)
  - Referencias con enlace directo a cada ficha técnica utilizada
  - Conversación con memoria de contexto

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

3. Configure la API key:

   Cree un archivo `.env` en el directorio raíz y añada su API key:
   ```
   OPENAI_API_KEY=su_api_key_openai
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
3. Revise la respuesta, la traza de recuperación y las referencias a las fichas técnicas oficiales

### Ajustes Adicionales

- **Visualización de razonamiento**: Puede activar o desactivar la visualización del proceso de recuperación en la barra lateral
- **Búsqueda avanzada**: Para las formulaciones magistrales, puede elegir entre el método de búsqueda estándar o avanzado

## Tecnologías

- **Streamlit**: Framework para la interfaz de usuario
- **OpenAI API (GPT-4o mini)**: Redacción de formulaciones, prospectos y respuestas
- **CIMA REST API**: API oficial de la AEMPS (medicamentos, maestras, buscarEnFichaTecnica, docSegmentado)
- **Pydantic**: Estado tipado del grafo RAG y validación de resultados
- **Python / aiohttp**: Cliente HTTP asíncrono

## Requisitos

- Python 3.8 o superior
- Conexión a Internet para acceder a las APIs
- API key de OpenAI

## Configuración en Streamlit Cloud

Para implementar en Streamlit Cloud, agregue el siguiente secret:
- OPENAI_API_KEY

## Nota Legal

Esta aplicación proporciona información con fines educativos e informativos. No reemplaza la consulta médica profesional. Todas las formulaciones magistrales deben ser revisadas por un farmacéutico cualificado antes de su elaboración.

La información mostrada proviene de la API CIMA de la AEMPS. Esta aplicación no está afiliada oficialmente con la AEMPS.
