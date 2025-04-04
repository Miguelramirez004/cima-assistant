import streamlit as st
import asyncio
import nest_asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from openai import AsyncOpenAI
import re
import os
from datetime import datetime

# Import agent modules
from formulacion import FormulationAgent
from config import Config
from openai_client import create_async_openai_client
from perplexity_client import PerplexityClient
from prospecto import ProspectoGenerator  # Add import for ProspectoGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure page settings
st.set_page_config(page_title="CIMA Assistant", layout="wide")

# Global executor for running async code
executor = ThreadPoolExecutor(max_workers=4)

# Ensure we have a single, reusable event loop
@st.cache_resource
def get_event_loop():
    """Get a reusable event loop with proper error handling"""
    try:
        # First try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            # Create a new one if closed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        # Create a new loop if there isn't one in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop

# Improved run_async function to handle event loop issues
def run_async(async_func, *args, **kwargs):
    """Run an async function in a dedicated event loop with proper cleanup"""
    def run_in_executor():
        # Create a new event loop for this specific operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_func(*args, **kwargs))
        except Exception as e:
            logger.error(f"Error in async execution: {str(e)}")
            raise
        finally:
            # Give pending tasks a chance to complete
            pending = asyncio.all_tasks(loop)
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
            # Close the loop properly
            if hasattr(loop, 'shutdown_asyncgens'):
                loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
    
    # Run the async code in a separate thread with its own event loop
    return executor.submit(run_in_executor).result()

# Initialize resources with proper lifecycle management
@st.cache_resource
def init_resources():
    """Initialize OpenAI client and agents with proper async resource management"""
    openai_client = create_async_openai_client(api_key=Config.OPENAI_API_KEY)
    
    # Initialize the formulation agent
    formulation_agent = FormulationAgent(openai_client)
    
    # Initialize Perplexity client for CIMA consultations
    perplexity_client = PerplexityClient(api_key=Config.PERPLEXITY_API_KEY)
    
    # Initialize prospecto generator
    prospecto_generator = ProspectoGenerator(openai_client)
    
    # Register cleanup handler for Streamlit session end
    def cleanup_resources():
        """Properly clean up resources when the Streamlit session ends"""
        try:
            # Run the close methods in a new event loop
            cleanup_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(cleanup_loop)
            
            # Close agents that have session resources
            try:
                cleanup_loop.run_until_complete(asyncio.gather(
                    formulation_agent.close(),
                    prospecto_generator.close(),
                    return_exceptions=True
                ))
            except Exception as e:
                logger.error(f"Error closing agents: {str(e)}")
                
            cleanup_loop.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    # Register the cleanup function to be called on app shutdown
    import atexit
    atexit.register(cleanup_resources)
    
    return formulation_agent, perplexity_client, prospecto_generator

# Custom CSS with just the essential styling
st.markdown("""
<style>
    /* Apple-style font for entire app */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
    }
    
    .main .block-container {padding-top: 2rem;}
    .stTabs [data-baseweb="tab-panel"] {padding-top: 1rem;}
    div.stButton > button:first-child {background-color: #4CAF50; color: white;}
    div.stButton > button:hover {background-color: #45a049;}
    
    /* Info box styling */
    .info-box {
        background-color: #2E7D32;
        border-left: 6px solid #1B5E20;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* Reasoning box styling */
    .reasoning-box {
        background-color: #f8f9fa;
        border-left: 6px solid #10a37f;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 4px;
    }
    
    /* References styling */
    .reference-item {
        background-color: #f0f2f6;
        border-left: 4px solid #4b5f84;
        padding: 10px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 0.9em;
    }
    
    .reference-title {
        font-weight: bold;
        color: #2c3e50;
    }
    
    .reference-url {
        color: #3498db;
        word-break: break-all;
    }
    
    /* Hide Streamlit elements for cleaner UI */
    .css-18e3th9 {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Better spacing for chat messages */
    [data-testid="stChatMessage"] {
        margin-bottom: 0.5rem;
    }
    
    /* Animated thinking indicator */
    @keyframes thinking-animation {
        0% { opacity: 0.4; }
        50% { opacity: 1.0; }
        100% { opacity: 0.4; }
    }
    
    .thinking-indicator {
        animation: thinking-animation 1.5s infinite;
        background-color: #f0f2f6;
        border-left: 6px solid #3498db;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 4px;
    }
    
    /* Debug info */
    .debug-info {
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        padding: 10px;
        margin-top: 10px;
        font-size: 0.8em;
        font-family: monospace;
        white-space: pre-wrap;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables more efficiently
if 'resources' not in st.session_state:
    st.session_state.resources = init_resources()
    st.session_state.chat_history = []
    st.session_state.formulation_history = []
    st.session_state.prospecto_history = []  # Add prospecto history
    st.session_state.search_history = set()
    st.session_state.messages = []
    st.session_state.current_query = ""
    st.session_state.use_langgraph = True  # Default to using the improved search
    st.session_state.show_reasoning = True  # New setting for showing reasoning
    st.session_state.debug_mode = False    # Debug mode

# Title and description
st.title("🧪 CIMA Assistant")
st.markdown("### *Sistema inteligente de consulta para formulación magistral y CIMA*")

# Sidebar 
with st.sidebar:
    st.header("Información")
    st.markdown("""
    Este asistente utiliza la API CIMA (Centro de Información online de Medicamentos) de la AEMPS para proporcionar:
    
    - Formulaciones magistrales detalladas
    - Consultas sobre medicamentos con IA avanzada
    - Prospectos de medicamentos
    - Referencias a información oficial
    """)
    
    # Add search mode setting for formulation (still using LangGraph there)
    st.header("Ajustes")
    use_langgraph = st.toggle("Usar búsqueda avanzada para formulación", value=st.session_state.use_langgraph)
    
    # Update session state and agents if toggle changed
    if use_langgraph != st.session_state.use_langgraph:
        st.session_state.use_langgraph = use_langgraph
        # Update formulation agent settings
        if st.session_state.resources and st.session_state.resources[0]:
            st.session_state.resources[0].use_langgraph = use_langgraph
        st.info(f"Modo de búsqueda para formulación: {'Avanzado' if use_langgraph else 'Estándar'}")
    
    # Add toggle for showing reasoning process
    show_reasoning = st.toggle("Mostrar proceso de razonamiento", value=st.session_state.show_reasoning)
    if show_reasoning != st.session_state.show_reasoning:
        st.session_state.show_reasoning = show_reasoning
        st.info(f"Visualización de razonamiento: {'Activado' if show_reasoning else 'Desactivado'}")
    
    # Debug mode toggle (hidden in production, enable with query param ?debug=true)
    debug_param = st.query_params.get("debug")
    if debug_param == "true":
        debug_mode = st.toggle("Modo depuración", value=st.session_state.debug_mode)
        if debug_mode != st.session_state.debug_mode:
            st.session_state.debug_mode = debug_mode
            st.info(f"Modo depuración: {'Activado' if debug_mode else 'Desactivado'}")
    
    st.header("Historial de búsquedas")
    if st.session_state.search_history:
        for query in list(st.session_state.search_history)[-5:]:
            st.markdown(f"- {query}")
    else:
        st.markdown("No hay búsquedas recientes")
    
    if st.button("Limpiar historial"):
        # Reset state
        st.session_state.search_history = set()
        st.session_state.formulation_history = []
        st.session_state.prospecto_history = []  # Clear prospecto history
        if st.session_state.resources and st.session_state.resources[1]:
            # Clear Perplexity client history
            st.session_state.resources[1].clear_history()
        st.session_state.messages = []
        st.rerun()

    # Add diagnostic button (keep for CIMA API testing)
    if st.button("Diagnóstico CIMA", key="diagnostico"):
        st.info("Ejecutando diagnóstico de conexión con CIMA...")
        
        progress = st.progress(0)
        status = st.empty()
        
        # Prueba 1: Conexión básica a CIMA
        status.text("Probando conexión básica a CIMA...")
        
        try:
            import requests
            response = requests.get("https://cima.aemps.es/cima/rest/medicamentos")
            if response.status_code == 200:
                st.success(f"✅ Conexión básica a CIMA exitosa (Status: {response.status_code})")
            else:
                st.error(f"❌ Error en conexión básica a CIMA (Status: {response.status_code})")
            progress.progress(25)
        except Exception as e:
            st.error(f"❌ Error en conexión básica: {str(e)}")
        
        # Prueba 2: Acceso a medicamento específico
        status.text("Probando acceso a medicamento específico...")
        
        try:
            # Usar MINOXIDIL BIORGA como prueba
            nregistro = "78929" # MINOXIDIL BIORGA
            response = requests.get(f"https://cima.aemps.es/cima/rest/medicamento?nregistro={nregistro}")
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ Acceso a medicamento exitoso: {data.get('nombre', 'Sin nombre')}")
            else:
                st.error(f"❌ Error en acceso a medicamento (Status: {response.status_code})")
            progress.progress(50)
        except Exception as e:
            st.error(f"❌ Error en acceso a medicamento: {str(e)}")
        
        # Prueba 3: Acceso a ficha técnica
        status.text("Probando acceso a ficha técnica...")
        
        try:
            response = requests.get(f"https://cima.aemps.es/cima/rest/docSegmentado/contenido/1?nregistro={nregistro}")
            if response.status_code == 200:
                st.success(f"✅ Acceso a ficha técnica exitoso via API")
            else:
                st.error(f"❌ Error en acceso a ficha técnica via API (Status: {response.status_code})")
            
            # Probar acceso directo HTML
            response = requests.get(f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html")
            if response.status_code == 200:
                st.success(f"✅ Acceso a ficha técnica HTML exitoso (longitud: {len(response.text)})")
            else:
                st.error(f"❌ Error en acceso a ficha técnica HTML (Status: {response.status_code})")
            progress.progress(75)
        except Exception as e:
            st.error(f"❌ Error en acceso a ficha técnica: {str(e)}")
        
        # Prueba 4: Impresión de respuesta completa para diagnóstico
        status.text("Obteniendo detalles completos...")
        
        try:
            url = f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html"
            response = requests.get(url)
            
            if response.status_code == 200:
                content = response.text
                with st.expander("Ver primeros 1000 caracteres de la respuesta"):
                    st.code(content[:1000])
                st.success(f"✅ Contenido obtenido correctamente (longitud: {len(content)})")
            else:
                st.error(f"❌ Error obteniendo contenido HTML (Status: {response.status_code})")
            progress.progress(100)
        except Exception as e:
            st.error(f"❌ Error obteniendo contenido: {str(e)}")
        
        status.empty()
        st.info("Diagnóstico completo")

# Main tabs - Updated to include Prospectos tab
tab1, tab2, tab3, tab4 = st.tabs(["Formulación Magistral", "Consultas CIMA", "Prospectos", "Historial"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("### Asistente para formulación magistral basado en CIMA")
        st.markdown("""
        <div class="info-box">
        Ingrese su consulta sobre formulación magistral. Especifique el principio activo, 
        concentración deseada y tipo de formulación para obtener mejores resultados.
        </div>
        """, unsafe_allow_html=True)
        
        # Handle query text area
        query_fm = st.text_area(
            "Ingrese su consulta sobre formulación:",
            value=st.session_state.current_query,
            height=100, 
            placeholder="Ejemplo: Suspensión de Ibuprofeno 100mg/ml para uso pediátrico"
        )
    
    with col2:
        st.write("### Ejemplos")
        example_queries = [
            "Suspensión de Omeprazol 2mg/ml",
            "Crema de Hidrocortisona al 1%",
            "Cápsulas de Melatonina 3mg",
            "Gel de Metronidazol 0.75%",
            "Solución de Minoxidil 5%"
        ]
        
        for example in example_queries:
            if st.button(example):
                st.session_state.current_query = example
                st.rerun()
        
        # Add a special example for the abacavir problem
        if st.button("MINOXIDIL BIORGA"):
            st.session_state.current_query = "Encontrar información sobre MINOXIDIL BIORGA"
            st.rerun()
    
    if st.button("Consultar Formulación", type="primary"):
        if not query_fm:
            st.warning("Por favor ingrese una consulta")
        else:
            # Check if query contains uppercase medication name like MINOXIDIL BIORGA
            uppercase_names = re.findall(r'\b[A-Z]{2,}\s+[A-Z]{2,}\b', query_fm.upper())
            if uppercase_names:
                st.info(f"⚠️ Se ha detectado un nombre específico de medicamento: {uppercase_names[0]}. Se realizará una búsqueda directa.")
            
            # Update current query
            st.session_state.current_query = query_fm
            
            # Add to search history
            st.session_state.search_history.add(query_fm)
            
            # Progress indicators for better user experience
            progress_placeholder = st.empty()
            status_text = st.empty()
            
            with st.spinner("Procesando su consulta..."):
                try:
                    # Show progress updates
                    with progress_placeholder.container():
                        progress_bar = st.progress(0)
                    
                    status_text.text("Buscando información en CIMA...")
                    progress_bar.progress(25)
                    
                    # Set the agent's search mode based on current setting
                    st.session_state.resources[0].use_langgraph = st.session_state.use_langgraph
                    
                    # Process response using our managed event loop
                    response = run_async(st.session_state.resources[0].answer_question(query_fm))
                    
                    # Update progress
                    status_text.text("Generando formulación...")
                    progress_bar.progress(75)
                    
                    # Store in formulation history
                    st.session_state.formulation_history.append({
                        "query": query_fm,
                        "response": response["answer"],
                        "context": response["context"],
                        "references": response["references"]
                    })
                    
                    # Complete progress
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_placeholder.empty()
                    
                    st.subheader("Formulación:")
                    st.markdown(response["answer"])
                    
                    # Extract and display references
                    references = re.findall(r'\[Ref \d+:.*?\]', response["answer"])
                    if references:
                        st.subheader("Referencias utilizadas:")
                        for ref in references:
                            st.markdown(f"- {ref}")
                    
                    with st.expander("Ver contexto de CIMA"):
                        st.markdown(response["context"])
                    
                    # Option to download the formulación
                    formulacion_text = f"""# Formulación Magistral

## Consulta
{query_fm}

## Formulación
{response["answer"]}

## Referencias
{response["context"]}
"""
                    st.download_button(
                        label="Descargar formulación",
                        data=formulacion_text,
                        file_name=f"formulacion_{query_fm[:30].replace(' ', '_')}.md",
                        mime="text/markdown"
                    )
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Error processing formulation query: {str(e)}")

with tab2:
    st.write("### Chat con experto CIMA")
    st.markdown("""
    <div class="info-box">
    Realice consultas sobre medicamentos.
    Puede preguntar sobre indicaciones, contraindicaciones, dosis, efectos secundarios, etc.
    </div>
    """, unsafe_allow_html=True)
    
    # Example section
    with st.expander("Ver ejemplos de consultas"):
        st.markdown("""
        - ¿Cuáles son los efectos secundarios del ibuprofeno?
        - ¿Qué dosis de paracetamol es segura para niños?
        - ¿Qué interacciones tiene la simvastatina con otros medicamentos?
        - ¿Cuáles son las contraindicaciones del omeprazol?
        - ¿Es seguro tomar metformina durante el embarazo?
        - ¿Cuál es la diferencia entre lorazepam y diazepam?
        """)
    
    # Chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # If the message has reasoning and references, display structured content
                if message["role"] == "assistant" and "reasoning" in message and "references" in message:
                    
                    # Show reasoning if enabled
                    if st.session_state.show_reasoning and message["reasoning"]:
                        st.markdown("""<div class="reasoning-box">
                        <h4>💭 Proceso de Razonamiento</h4>
                        {reasoning}
                        </div>
                        """.format(reasoning=message["reasoning"]), unsafe_allow_html=True)
                    
                    # Show main answer
                    st.markdown(message["content"])
                    
                    # Show references
                    if message["references"] and len(message["references"]) > 0:
                        st.markdown("<h4>📚 Referencias</h4>", unsafe_allow_html=True)
                        for ref in message["references"]:
                            title = ref.get("title", "")
                            url = ref.get("url", "")
                            if url:
                                st.markdown(f"""<div class="reference-item">
                                <span class="reference-title">{title}</span><br>
                                <a href="{url}" target="_blank" class="reference-url">{url}</a>
                                </div>""", unsafe_allow_html=True)
                            else:
                                st.markdown(f"""<div class="reference-item">
                                <span class="reference-title">{title}</span>
                                </div>""", unsafe_allow_html=True)
                else:
                    # Regular message display
                    st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Escriba su consulta sobre medicamentos..."):
        # Add to search history
        st.session_state.search_history.add(prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process and display assistant response using Perplexity
        with st.chat_message("assistant"):
            # Progress indicators
            progress_placeholder = st.empty()
            
            # Create a placeholder for the "thinking" animation
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("""
            <div class="thinking-indicator">
            <h4>💭 Pensando...</h4>
            <p>Estoy analizando la información médica disponible sobre su consulta. Este proceso puede tomar unos segundos...</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Buscando información..."):
                try:
                    # Get the perplexity client
                    perplexity_client = st.session_state.resources[1]
                    
                    # Process query with Perplexity
                    try:
                        # Try using async method
                        response = run_async(perplexity_client.ask_cima_question_async, prompt)
                    except Exception as async_error:
                        # Fall back to sync method if async fails
                        logger.warning(f"Async Perplexity call failed, falling back to sync: {str(async_error)}")
                        response = perplexity_client.ask_cima_question(prompt)
                    
                    # Clear the thinking animation
                    thinking_placeholder.empty()
                    
                    # Show debug info if enabled
                    if st.session_state.debug_mode:
                        st.markdown("""<div class="debug-info">
                        Raw response length: {length}
                        Has 'answer': {has_answer}
                        Has 'reasoning': {has_reasoning}
                        Has 'references': {has_references}
                        References count: {ref_count}
                        </div>""".format(
                            length=len(response.get("full_content", "")),
                            has_answer="Yes" if response.get("answer") else "No",
                            has_reasoning="Yes" if response.get("reasoning") else "No", 
                            has_references="Yes" if response.get("references") else "No",
                            ref_count=len(response.get("references", []))
                        ), unsafe_allow_html=True)
                    
                    # Extract structured data from response
                    reasoning = response.get("reasoning", "")
                    answer = response.get("answer", "")
                    references = response.get("references", [])
                    
                    # Ensure we have a valid answer (fallback to full content if needed)
                    if not answer and "full_content" in response:
                        answer = response["full_content"]
                        # Add a note about parsing issues
                        if "full_content" in response and response["full_content"]:
                            answer = "**Nota:** Hubo un problema al estructurar la respuesta, pero aquí está la información:\n\n" + answer
                    
                    # Show reasoning if enabled
                    if st.session_state.show_reasoning and reasoning:
                        st.markdown("""<div class="reasoning-box">
                        <h4>💭 Proceso de Razonamiento</h4>
                        {reasoning}
                        </div>
                        """.format(reasoning=reasoning), unsafe_allow_html=True)
                    
                    # Show the main answer
                    st.markdown(answer)
                    
                    # Show references
                    if references and len(references) > 0:
                        st.markdown("<h4>📚 Referencias</h4>", unsafe_allow_html=True)
                        for ref in references:
                            title = ref.get("title", "")
                            url = ref.get("url", "")
                            if url:
                                st.markdown(f"""<div class="reference-item">
                                <span class="reference-title">{title}</span><br>
                                <a href="{url}" target="_blank" class="reference-url">{url}</a>
                                </div>""", unsafe_allow_html=True)
                            else:
                                st.markdown(f"""<div class="reference-item">
                                <span class="reference-title">{title}</span>
                                </div>""", unsafe_allow_html=True)
                    
                    # Create a full message with all components for history
                    full_message = {
                        "role": "assistant", 
                        "content": answer,
                        "reasoning": reasoning,
                        "references": references
                    }
                    
                    # Add to session state
                    st.session_state.messages.append(full_message)
                    
                except Exception as e:
                    error_message = f"Lo siento, ha ocurrido un error al procesar su consulta: {str(e)}"
                    st.markdown(error_message)
                    logger.error(f"Error in Perplexity chat response: {str(e)}")
                    # Add error message to session state so conversation continues
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Button for new conversation
    if st.button("Nueva conversación", key="new_chat"):
        st.session_state.messages = []
        # Clear Perplexity history
        if st.session_state.resources and st.session_state.resources[1]:
            st.session_state.resources[1].clear_history()
        st.rerun()

# Add new Prospectos tab implementation
with tab3:
    st.write("### Generador de Prospectos de Medicamentos")
    st.markdown("""
    <div class="info-box">
    Genere prospectos completos para medicamentos según la normativa de la AEMPS.
    Especifique el nombre del medicamento o principio activo para obtener mejores resultados.
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for input and examples
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Input for prospecto query
        prospecto_query = st.text_area(
            "Solicitud para generar un prospecto:",
            value="",
            height=100,
            placeholder="Ejemplo: Generar prospecto para Ibuprofeno 600mg"
        )
    
    with col2:
        st.write("### Ejemplos")
        example_queries = [
            "Generar prospecto para Ibuprofeno 600mg",
            "Prospecto de Omeprazol 20mg",
            "Crear prospecto para Amoxicilina 500mg",
            "Prospecto para MINOXIDIL BIORGA"
        ]
        
        for example in example_queries:
            if st.button(example, key=f"prospecto_{example}"):
                # Set the example as the current query
                prospecto_query = example
                st.rerun()
    
    # Generate button
    if st.button("Generar Prospecto", type="primary", key="generate_prospecto"):
        if not prospecto_query:
            st.warning("Por favor ingrese una consulta")
        else:
            # Add to search history
            st.session_state.search_history.add(prospecto_query)
            
            # Progress indicators
            progress_placeholder = st.empty()
            status_text = st.empty()
            
            with st.spinner("Generando prospecto..."):
                try:
                    # Show progress updates
                    with progress_placeholder.container():
                        progress_bar = st.progress(0)
                    
                    status_text.text("Buscando información en CIMA...")
                    progress_bar.progress(30)
                    
                    # Get prospecto generator from resources
                    prospecto_generator = st.session_state.resources[2]  # Third element in the tuple
                    
                    # Generate prospecto
                    response = run_async(prospecto_generator.generate_prospecto, prospecto_query)
                    
                    # Update progress
                    status_text.text("Finalizando prospecto...")
                    progress_bar.progress(80)
                    
                    # Add to history
                    st.session_state.prospecto_history.append({
                        "query": prospecto_query,
                        "prospecto": response["prospecto"],
                        "context": response["context"],
                        "medication": response["medication"]
                    })
                    
                    # Complete progress
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_placeholder.empty()
                    
                    # Display the prospecto
                    st.subheader(f"Prospecto para: {response['medication']}")
                    st.markdown(response["prospecto"])
                    
                    # Option to download
                    prospecto_text = f"""# PROSPECTO: INFORMACIÓN PARA EL USUARIO

{response["prospecto"]}

---
Generado para: {response["medication"]}
Fecha de generación: {datetime.now().strftime("%d/%m/%Y")}
"""
                    st.download_button(
                        label="Descargar prospecto",
                        data=prospecto_text,
                        file_name=f"prospecto_{response['medication'].replace(' ', '_')[:30]}.md",
                        mime="text/markdown"
                    )
                    
                    # Show context in expandable section
                    with st.expander("Ver datos utilizados de CIMA"):
                        st.markdown(response["context"])
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Error generating prospecto: {str(e)}")

# Update the Historial tab with subtabs
with tab4:
    # Create subtabs for different history types
    hist_tab1, hist_tab2 = st.tabs(["Formulaciones", "Prospectos"])
    
    with hist_tab1:
        st.header("Historial de formulaciones")
        
        if not st.session_state.formulation_history:
            st.info("No hay formulaciones en el historial")
        else:
            for i, item in enumerate(st.session_state.formulation_history):
                with st.expander(f"Formulación: {item['query']}"):
                    st.markdown(item["response"])
                    st.download_button(
                        label="Descargar",
                        data=f"""# Formulación Magistral\n\n## Consulta\n{item['query']}\n\n## Formulación\n{item["response"]}\n\n## Referencias\n{item["context"]}""",
                        file_name=f"formulacion_{i}.md",
                        mime="text/markdown"
                    )
    
    with hist_tab2:
        st.header("Historial de prospectos")
        
        if not st.session_state.prospecto_history:
            st.info("No hay prospectos en el historial")
        else:
            for i, item in enumerate(st.session_state.prospecto_history):
                with st.expander(f"Prospecto: {item['medication']}"):
                    st.markdown(item["prospecto"])
                    st.download_button(
                        label="Descargar",
                        data=f"""# PROSPECTO: INFORMACIÓN PARA EL USUARIO\n\n{item["prospecto"]}\n\n---\nGenerado para: {item["medication"]}\nFecha de generación: {datetime.now().strftime("%d/%m/%Y")}""",
                        file_name=f"prospecto_{item['medication'].replace(' ', '_')[:30]}_{i}.md",
                        mime="text/markdown"
                    )
