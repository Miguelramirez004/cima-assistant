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
from formulacion import FormulationAgent, CIMAExpertAgent
from config import Config
from openai_client import create_async_openai_client

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

# Initialize async resources with proper lifecycle management
@st.cache_resource
def init_agents():
    """Initialize OpenAI client and agents with proper async resource management"""
    openai_client = create_async_openai_client(api_key=Config.OPENAI_API_KEY)
    
    # Initialize the agents with the client
    formulation_agent = FormulationAgent(openai_client)
    cima_expert_agent = CIMAExpertAgent(openai_client)
    
    # Register cleanup handler for Streamlit session end
    def cleanup_resources():
        """Properly clean up resources when the Streamlit session ends"""
        try:
            # Run the close methods in a new event loop
            cleanup_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(cleanup_loop)
            cleanup_loop.run_until_complete(asyncio.gather(
                formulation_agent.close(), 
                cima_expert_agent.close(),
                return_exceptions=True
            ))
            cleanup_loop.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    # Register the cleanup function to be called on app shutdown
    import atexit
    atexit.register(cleanup_resources)
    
    return formulation_agent, cima_expert_agent

# Custom CSS with just the essential styling
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    .stTabs [data-baseweb="tab-panel"] {padding-top: 1rem;}
    div.stButton > button:first-child {background-color: #4CAF50; color: white;}
    div.stButton > button:hover {background-color: #45a049;}
    
    /* Changed info-box background to light gray */
    .info-box {
        background-color: #2E7D32;
        border-left: 6px solid #1B5E20;
        padding: 10px;
        margin-bottom: 10px;
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
</style>
""", unsafe_allow_html=True)

# Initialize session state variables more efficiently
if 'agents' not in st.session_state:
    st.session_state.agents = init_agents()
    st.session_state.chat_history = []
    st.session_state.formulation_history = []
    st.session_state.search_history = set()
    st.session_state.messages = []
    st.session_state.current_query = ""

# Title and description
st.title("🧪 CIMA Assistant")
st.markdown("### *Sistema inteligente de consulta para formulación magistral y CIMA*")

# Sidebar 
with st.sidebar:
    st.header("Información")
    st.markdown("""
    Este asistente utiliza la API CIMA (Centro de Información online de Medicamentos) de la AEMPS para proporcionar:
    
    - Formulaciones magistrales detalladas
    - Consultas sobre medicamentos
    - Referencias directas a fichas técnicas
    """)
    
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
        if st.session_state.agents and st.session_state.agents[1]:
            # Add a clear_history method if it doesn't exist
            if hasattr(st.session_state.agents[1], 'clear_history'):
                st.session_state.agents[1].clear_history()
            else:
                # If no clear_history method, clear conversation_history directly
                st.session_state.agents[1].conversation_history = []
        st.session_state.messages = []
        st.rerun()

    # Add diagnostic button
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

# Main tabs
tab1, tab2, tab3 = st.tabs(["Formulación Magistral", "Consultas CIMA", "Historial"])

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
                    
                    # Process response using our managed event loop
                    response = run_async(st.session_state.agents[0].answer_question(query_fm))
                    
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
    Realice consultas sobre medicamentos registrados en CIMA. 
    Puede preguntar sobre indicaciones, contraindicaciones, efectos adversos, etc.
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Escriba su consulta sobre medicamentos..."):
        # Add to search history
        st.session_state.search_history.add(prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Check if prompt contains uppercase medication name like MINOXIDIL BIORGA
        uppercase_names = re.findall(r'\b[A-Z]{2,}\s+[A-Z]{2,}\b', prompt.upper())
        if uppercase_names:
            info_msg = st.empty()
            info_msg.info(f"⚠️ Se ha detectado un nombre específico de medicamento: {uppercase_names[0]}. Se realizará una búsqueda directa.")

        # Process and display assistant response
        with st.chat_message("assistant"):
            # Progress indicators
            progress_placeholder = st.empty()
            status_text = st.empty()
            
            with st.spinner("Buscando información en CIMA..."):
                try:
                    # Show progress updates
                    with progress_placeholder.container():
                        progress_bar = st.progress(0)
                    
                    status_text.text("Consultando CIMA...")
                    progress_bar.progress(30)
                    
                    # Process response using our managed event loop
                    response = run_async(st.session_state.agents[1].chat(prompt))
                    
                    # Update progress
                    status_text.text("Generando respuesta...")
                    progress_bar.progress(80)
                    
                    # Clear progress indicators
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_placeholder.empty()
                    if uppercase_names:
                        info_msg.empty()  # Remove info message if present
                    
                    # Show response
                    st.markdown(response["answer"])
                    
                    # Show sources in expander
                    with st.expander("Ver fuentes"):
                        st.markdown(response["context"])
                        
                    # Add to session state (MOVED INSIDE TRY BLOCK)
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                    
                except Exception as e:
                    st.markdown(f"Error: {str(e)}")
                    logger.error(f"Error in chat response: {str(e)}")
                    # Add error message to session state so conversation continues
                    error_message = f"Lo siento, ha ocurrido un error al procesar su consulta: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Button for new conversation
    if st.button("Nueva conversación", key="new_chat"):
        st.session_state.messages = []
        # Make sure clear_history exists before calling it
        if hasattr(st.session_state.agents[1], 'clear_history'):
            st.session_state.agents[1].clear_history()
        else:
            # If no clear_history method, clear conversation_history directly
            st.session_state.agents[1].conversation_history = []
        st.rerun()

with tab3:
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