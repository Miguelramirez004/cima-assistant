import streamlit as st
import asyncio
import nest_asyncio
import openai
from openai import AsyncOpenAI
import re
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from formulacion import FormulationAgent, CIMAExpertAgent
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (for local development)
load_dotenv()

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure page
st.set_page_config(page_title="CIMA Assistant", layout="wide")

# Global executor for running async code
executor = ThreadPoolExecutor(max_workers=4)

# Simple CSS
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    .stTabs [data-baseweb="tab-panel"] {padding-top: 1rem;}
    div.stButton > button:first-child {background-color: #4CAF50; color: white;}
    div.stButton > button:hover {background-color: #45a049;}
    
    .info-box {
        background-color: #2E7D32;
        border-left: 6px solid #1B5E20;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

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

# Global OpenAI client for reuse
@st.cache_resource
def get_openai_client():
    """Get OpenAI client with proper API key handling"""
    # Try to get API key from Streamlit secrets first (for cloud deployment)
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Fall back to environment variables or Config
        api_key = os.getenv("OPENAI_API_KEY") or Config.OPENAI_API_KEY
    
    if not api_key:
        st.error("No se ha encontrado la API key de OpenAI. Verifique los secretos de Streamlit, variables de entorno o el archivo config.py")
        return None
        
    return AsyncOpenAI(api_key=api_key)

# Initialize and manage agent resources
@st.cache_resource
def init_agents():
    """Initialize the agents with proper resource management"""
    openai_client = get_openai_client()
    if not openai_client:
        return None, None
    
    formulation_agent = FormulationAgent(openai_client)
    cima_agent = CIMAExpertAgent(openai_client)
    
    # Register cleanup handler for Streamlit session end
    def cleanup_resources():
        """Properly clean up resources when the Streamlit session ends"""
        try:
            # Run the close methods in a new event loop
            cleanup_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(cleanup_loop)
            cleanup_loop.run_until_complete(asyncio.gather(
                formulation_agent.close(), 
                cima_agent.close(),
                return_exceptions=True
            ))
            cleanup_loop.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    # Register the cleanup function to be called on app shutdown
    import atexit
    atexit.register(cleanup_resources)
    
    return formulation_agent, cima_agent

# Initialize session state variables if not already present
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'formulation_history' not in st.session_state:
    st.session_state.formulation_history = []
if 'search_history' not in st.session_state:
    st.session_state.search_history = set()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'agents' not in st.session_state:
    st.session_state.agents = init_agents()

# Check OpenAI API key at startup
openai_client = get_openai_client()
if openai_client:
    st.success("✅ Conexión a OpenAI configurada correctamente")
else:
    st.error("❌ Error: No se pudo establecer conexión con OpenAI. Por favor configure la API key en los secretos de Streamlit.")
    
# Title
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
        st.session_state.search_history = set()
        st.session_state.formulation_history = []
        if st.session_state.agents and st.session_state.agents[1]:
            st.session_state.agents[1].clear_history()
        st.session_state.messages = []
        st.rerun()

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
                    
                    # Get formulation agent from session state
                    formulation_agent = st.session_state.agents[0]
                    if not formulation_agent:
                        st.error("No se puede conectar con OpenAI. Verifique su API key.")
                    else:
                        # Get response using our improved helper function
                        response = run_async(formulation_agent.answer_question, query_fm)
                        
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
                    
                    # Get CIMA agent from session state
                    cima_agent = st.session_state.agents[1]
                    if not cima_agent:
                        st.error("No se puede conectar con OpenAI. Verifique su API key.")
                    else:
                        # Process response using our improved helper function
                        response = run_async(cima_agent.chat, prompt)
                        
                        # Update progress
                        status_text.text("Generando respuesta...")
                        progress_bar.progress(80)
                        
                        # Clear progress indicators
                        progress_bar.progress(100)
                        status_text.empty()
                        progress_placeholder.empty()
                        
                        # Show response
                        st.markdown(response["answer"])
                        
                        # Show sources in expander
                        with st.expander("Ver fuentes"):
                            st.markdown(response["context"])
                except Exception as e:
                    logger.error(f"Error in chat response: {str(e)}")
                    st.markdown(f"Error: {str(e)}")
                    
        # Add to session state
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    
    # Button for new conversation
    if st.button("Nueva conversación", key="new_chat"):
        st.session_state.messages = []
        if st.session_state.agents and st.session_state.agents[1]:
            st.session_state.agents[1].clear_history()
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