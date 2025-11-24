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

# Professional Compact Dashboard CSS
st.markdown("""
<style>
    /* Professional font stack */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Roboto, sans-serif !important;
    }

    /* Compact spacing - reduce vertical gaps */
    .main .block-container {
        padding-top: 0.75rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    .stTabs [data-baseweb="tab-panel"] {padding-top: 0.75rem;}

    /* Reduce element spacing */
    .element-container {margin-bottom: 0.4rem;}

    /* Compact status panel */
    .status-panel {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 0.75rem;
        color: white;
        font-size: 0.85em;
    }

    .status-panel h4 {
        margin: 0 0 8px 0;
        font-size: 0.9em;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .status-log {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 6px;
        padding: 8px 10px;
        max-height: 180px;
        overflow-y: auto;
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 0.8em;
    }

    .status-message {
        padding: 3px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: flex-start;
        gap: 6px;
    }

    .status-message:last-child {border-bottom: none;}

    .status-timestamp {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.8em;
        min-width: 50px;
    }

    .status-text {
        flex: 1;
        line-height: 1.2;
    }

    /* Compact professional buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
        margin-top: 0.3rem;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }

    /* Compact info box */
    .info-box {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        border-left: 4px solid #6366F1;
        padding: 8px 12px;
        margin: 0.4rem 0;
        border-radius: 6px;
        font-size: 0.85em;
        color: #4338CA;
        line-height: 1.3;
    }

    /* Compact reasoning box */
    .reasoning-box {
        background: #FFFBEB;
        border-left: 3px solid #F59E0B;
        padding: 8px 12px;
        margin: 0.4rem 0;
        border-radius: 6px;
    }

    .reasoning-box h4 {
        font-size: 0.85em;
        color: #92400E;
        margin: 0 0 4px 0;
        font-weight: 600;
    }

    /* Compact professional references */
    .reference-item {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.8em;
        transition: all 0.2s;
    }

    .reference-item:hover {
        border-color: #9CA3AF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .reference-title {
        font-weight: 600;
        color: #111827;
        font-size: 0.85em;
        margin-bottom: 2px;
    }

    .reference-url {
        color: #6366F1;
        word-break: break-all;
        font-size: 0.8em;
    }

    /* Compact activity indicator */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    .activity-indicator {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        border-left: 3px solid #6366F1;
        padding: 8px 12px;
        margin-bottom: 0.5rem;
        border-radius: 6px;
        animation: pulse 2s ease-in-out infinite;
    }

    .activity-indicator h4 {
        font-size: 0.85em;
        color: #4338CA;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Compact text areas */
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #E5E7EB;
        padding: 8px;
        font-size: 0.9em;
    }

    .stTextArea textarea:focus {
        border-color: #6366F1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }

    /* Compact expander */
    .streamlit-expanderHeader {
        font-size: 0.85em;
        font-weight: 500;
        padding: 6px 10px;
        border-radius: 6px;
        background: #F9FAFB;
    }

    .streamlit-expanderContent {
        padding: 6px 0;
    }

    /* Prospecto container */
    .prospecto-container {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 16px;
        margin: 0.5rem 0;
        line-height: 1.6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Compact chat styling */
    [data-testid="stChatMessage"] {
        margin-bottom: 0.6rem;
        padding: 0.75rem;
        border-radius: 8px;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Professional compact tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #F9FAFB;
        padding: 4px;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 7px 14px;
        font-weight: 500;
        font-size: 0.9em;
    }

    /* Compact sidebar */
    [data-testid="stSidebar"] {
        background: #F9FAFB;
    }

    /* Reduce column gaps */
    .row-widget.stHorizontal {
        gap: 0.5rem;
    }

    /* Success/Error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 6px;
        padding: 0.6rem;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables more efficiently
if 'resources' not in st.session_state:
    st.session_state.resources = init_resources()
    st.session_state.chat_history = []
    st.session_state.formulation_history = []
    st.session_state.prospecto_history = []
    st.session_state.search_history = set()
    st.session_state.messages = []
    st.session_state.current_query = ""
    st.session_state.current_prospecto_query = ""
    st.session_state.use_langgraph = True
    st.session_state.show_reasoning = True
    st.session_state.debug_mode = False
    st.session_state.status_log = []  # New: Track thought process
    st.session_state.show_status = True  # New: Toggle for status display

# Helper function to add status messages
def add_status(message, icon="🔄"):
    """Add a timestamped status message to show the thought process"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.status_log.append({
        "time": timestamp,
        "message": message,
        "icon": icon
    })
    # Keep only last 20 messages
    if len(st.session_state.status_log) > 20:
        st.session_state.status_log = st.session_state.status_log[-20:]

def display_status_panel():
    """Display the status panel showing the thought process"""
    if st.session_state.status_log and st.session_state.show_status:
        log_html = ""
        for entry in st.session_state.status_log:
            log_html += f"""
            <div class="status-message">
                <span class="status-icon">{entry['icon']}</span>
                <span class="status-timestamp">{entry['time']}</span>
                <span class="status-text">{entry['message']}</span>
            </div>
            """

        st.markdown(f"""
        <div class="status-panel">
            <h4>🧠 Proceso de Pensamiento</h4>
            <div class="status-log">
                {log_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Clean title
st.title("🧪 CIMA Assistant")
st.caption("Sistema de consulta para formulación magistral y medicamentos")

# Professional Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuración")

    # Settings toggles
    show_status = st.toggle("🧠 Mostrar pensamiento", value=st.session_state.show_status,
                             help="Muestra el proceso interno de razonamiento del sistema")
    if show_status != st.session_state.show_status:
        st.session_state.show_status = show_status

    use_langgraph = st.toggle("🔍 Búsqueda avanzada", value=st.session_state.use_langgraph,
                               help="Activa búsqueda avanzada para formulaciones")
    if use_langgraph != st.session_state.use_langgraph:
        st.session_state.use_langgraph = use_langgraph
        if st.session_state.resources and st.session_state.resources[0]:
            st.session_state.resources[0].use_langgraph = use_langgraph

    show_reasoning = st.toggle("💡 Razonamiento IA", value=st.session_state.show_reasoning,
                                help="Muestra el razonamiento detallado de las respuestas")
    if show_reasoning != st.session_state.show_reasoning:
        st.session_state.show_reasoning = show_reasoning

    # Debug mode (hidden unless query param is set)
    debug_param = st.query_params.get("debug")
    if debug_param == "true":
        debug_mode = st.toggle("🐛 Depuración", value=st.session_state.debug_mode)
        if debug_mode != st.session_state.debug_mode:
            st.session_state.debug_mode = debug_mode

    st.divider()

    # Recent searches
    if st.session_state.search_history:
        with st.expander("📝 Búsquedas recientes"):
            for query in list(st.session_state.search_history)[-5:]:
                st.caption(f"• {query}")

    st.divider()

    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.search_history = set()
            st.session_state.formulation_history = []
            st.session_state.prospecto_history = []
            st.session_state.status_log = []
            if st.session_state.resources and st.session_state.resources[1]:
                st.session_state.resources[1].clear_history()
            st.session_state.messages = []
            add_status("Sistema reiniciado", "🔄")
            st.rerun()

    with col2:
        if st.button("🔧 Test", use_container_width=True, key="diagnostico_btn"):
            st.session_state.show_diagnostico = not st.session_state.get('show_diagnostico', False)

    # Show diagnostic when toggled
    if st.session_state.get('show_diagnostico', False):
        with st.expander("🔧 Diagnóstico CIMA", expanded=True):
            import requests
            add_status("Ejecutando diagnóstico de conexión", "🔍")

            try:
                response = requests.get("https://cima.aemps.es/cima/rest/medicamentos", timeout=5)
                if response.status_code == 200:
                    st.success("✅ Conexión CIMA OK")
                    add_status("Conexión CIMA verificada", "✅")
                else:
                    st.error(f"❌ Error: {response.status_code}")
                    add_status(f"Error conexión: {response.status_code}", "❌")
            except Exception as e:
                st.error(f"❌ Error: {str(e)[:50]}")
                add_status("Error en diagnóstico", "❌")

            try:
                nregistro = "78929"
                response = requests.get(f"https://cima.aemps.es/cima/rest/medicamento?nregistro={nregistro}", timeout=5)
                if response.status_code == 200:
                    st.success("✅ Acceso medicamentos OK")
                    add_status("Acceso a medicamentos verificado", "✅")
                else:
                    st.error(f"❌ Error: {response.status_code}")
                    add_status(f"Error acceso medicamentos: {response.status_code}", "❌")
            except Exception as e:
                st.error(f"❌ Error: {str(e)[:50]}")
                add_status("Error verificando medicamentos", "❌")

# Main tabs - Updated to include Prospectos tab
tab1, tab2, tab3, tab4 = st.tabs(["Formulación Magistral", "Consultas CIMA", "Prospectos", "Historial"])

with tab1:
    st.markdown("### 💊 Formulación Magistral")

    # Display status panel at the top
    display_status_panel()

    # Professional info card
    st.markdown("""
    <div class="info-box">
    📋 Especifique el principio activo, concentración deseada y tipo de formulación para obtener una receta detallada.
    </div>
    """, unsafe_allow_html=True)

    # Query input
    query_fm = st.text_area(
        "Ingrese su consulta:",
        value=st.session_state.current_query,
        height=100,
        placeholder="Ejemplo: Suspensión de Ibuprofeno 100mg/ml para uso pediátrico"
    )

    # Collapsed examples
    with st.expander("💡 Ver ejemplos de consultas"):
        example_queries = [
            "Suspensión de Omeprazol 2mg/ml",
            "Crema de Hidrocortisona al 1%",
            "Cápsulas de Melatonina 3mg",
            "Gel de Metronidazol 0.75%",
            "Solución de Minoxidil 5%",
            "MINOXIDIL BIORGA"
        ]

        cols = st.columns(3)
        for i, example in enumerate(example_queries):
            with cols[i % 3]:
                if st.button(example, key=f"fm_ex_{i}", use_container_width=True):
                    st.session_state.current_query = example if example != "MINOXIDIL BIORGA" else "Encontrar información sobre MINOXIDIL BIORGA"
                    add_status(f"Ejemplo seleccionado: {example}", "📝")
                    st.rerun()
    
    if st.button("Consultar Formulación", type="primary"):
        if not query_fm:
            st.warning("Por favor ingrese una consulta")
        else:
            # Check if query contains uppercase medication name like MINOXIDIL BIORGA
            uppercase_names = re.findall(r'\b[A-Z]{2,}\s+[A-Z]{2,}\b', query_fm.upper())
            if uppercase_names:
                st.info(f"⚠️ Se ha detectado un nombre específico de medicamento: {uppercase_names[0]}. Se realizará una búsqueda directa.")
            
            # Update current query and add to history
            st.session_state.current_query = query_fm
            st.session_state.search_history.add(query_fm)

            # Add detailed status messages
            add_status(f"Nueva consulta recibida: {query_fm[:50]}...", "📥")

            # Processing with detailed status updates
            with st.spinner("🔄 Procesando consulta..."):
                try:
                    # Step 1: Configure search mode
                    add_status("Configurando modo de búsqueda...", "⚙️")
                    st.session_state.resources[0].use_langgraph = st.session_state.use_langgraph
                    mode_text = "avanzada" if st.session_state.use_langgraph else "estándar"
                    add_status(f"Modo de búsqueda: {mode_text}", "✓")

                    # Step 2: Searching CIMA database
                    add_status("Conectando con base de datos CIMA...", "🔍")
                    add_status("Buscando información del medicamento...", "📊")

                    # Process response
                    response = run_async(st.session_state.resources[0].answer_question(query_fm))

                    add_status("Información obtenida de CIMA", "✅")

                    # Check for prospecto redirect
                    if "use la pestaña 'Prospectos'" in response.get("answer", ""):
                        add_status("Detectada solicitud de prospecto - redirigiendo", "↪️")
                        st.info("💡 Esta consulta es para generar un prospecto. Usa la pestaña 'Prospectos'.")
                        st.session_state.current_prospecto_query = query_fm
                        st.rerun()

                    # Step 3: Process and format results
                    add_status("Procesando y formateando resultados...", "📝")

                    # Store in history
                    st.session_state.formulation_history.append({
                        "query": query_fm,
                        "response": response["answer"],
                        "context": response["context"],
                        "references": response["references"]
                    })

                    add_status("Formulación generada exitosamente", "✅")
                    
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
    st.markdown("### 💬 Consultas CIMA")

    # Display status panel
    display_status_panel()

    # Professional info card
    st.markdown("""
    <div class="info-box">
    💊 Realice preguntas sobre medicamentos: indicaciones, dosis, contraindicaciones, efectos secundarios, interacciones, etc.
    </div>
    """, unsafe_allow_html=True)

    # Example section
    with st.expander("💡 Ejemplos de preguntas"):
        examples = [
            "¿Cuáles son los efectos secundarios del ibuprofeno?",
            "¿Qué dosis de paracetamol es segura para niños?",
            "¿Qué interacciones tiene la simvastatina?",
            "¿Es seguro tomar metformina durante el embarazo?"
        ]
        for ex in examples:
            st.caption(f"• {ex}")
    
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
        # Add to history
        st.session_state.search_history.add(prompt)
        add_status(f"Nueva pregunta: {prompt[:50]}...", "💬")

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process and display assistant response
        with st.chat_message("assistant"):
            # Activity indicator
            activity_placeholder = st.empty()
            activity_placeholder.markdown("""
            <div class="activity-indicator">
            <h4>🤖 Analizando su consulta...</h4>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("🔍 Buscando información..."):
                try:
                    add_status("Iniciando análisis con IA...", "🤖")
                    add_status("Consultando fuentes médicas...", "📚")
                    # Get the perplexity client
                    perplexity_client = st.session_state.resources[1]

                    # Process query
                    try:
                        add_status("Ejecutando consulta avanzada...", "⚡")
                        response = run_async(perplexity_client.ask_cima_question_async, prompt)
                        add_status("Respuesta recibida de IA", "✅")
                    except Exception as async_error:
                        logger.warning(f"Async Perplexity call failed, falling back to sync: {str(async_error)}")
                        add_status("Reintentando con método alternativo...", "🔄")
                        response = perplexity_client.ask_cima_question(prompt)
                        add_status("Respuesta obtenida", "✅")

                    # Clear activity indicator
                    activity_placeholder.empty()
                    add_status("Formateando respuesta...", "📝")
                    
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

# Prospectos tab with status panel
with tab3:
    st.markdown("### 📄 Generador de Prospectos")

    # Display status panel
    display_status_panel()

    # Professional info card
    st.markdown("""
    <div class="info-box">
    📋 Genere prospectos completos según normativa AEMPS. Especifique el nombre del medicamento o principio activo.
    </div>
    """, unsafe_allow_html=True)

    # Input for prospecto query
    prospecto_query = st.text_area(
        "Solicitud de prospecto:",
        value=st.session_state.current_prospecto_query,
        height=100,
        placeholder="Ejemplo: Generar prospecto para Ibuprofeno 600mg"
    )

    # Collapsed examples
    with st.expander("💡 Ver ejemplos"):
        example_queries = [
            "Generar prospecto para Ibuprofeno 600mg",
            "Prospecto de Omeprazol 20mg",
            "Crear prospecto para Amoxicilina 500mg",
            "Prospecto para MINOXIDIL BIORGA"
        ]

        cols = st.columns(2)
        for i, example in enumerate(example_queries):
            with cols[i % 2]:
                if st.button(example, key=f"prospecto_ex_{i}", use_container_width=True):
                    st.session_state.current_prospecto_query = example
                    add_status(f"Ejemplo seleccionado: {example}", "📝")
                    st.rerun()
    
    # Generate button
    if st.button("Generar Prospecto", type="primary", key="generate_prospecto"):
        if not prospecto_query:
            st.warning("Por favor ingrese una consulta")
        else:
            # Add to search history
            st.session_state.search_history.add(prospecto_query)
            add_status(f"Solicitud de prospecto: {prospecto_query[:50]}...", "📄")

            # Processing with detailed status updates
            with st.spinner("🔄 Generando prospecto..."):
                try:
                    add_status("Inicializando generador de prospectos...", "⚙️")

                    # Get prospecto generator
                    prospecto_generator = st.session_state.resources[2]

                    add_status("Buscando información del medicamento en CIMA...", "🔍")
                    add_status("Extrayendo ficha técnica...", "📋")
                    add_status("Procesando indicaciones y contraindicaciones...", "📊")
                    add_status("Generando formato de prospecto AEMPS...", "📝")

                    # Generate prospecto
                    response = run_async(prospecto_generator.generate_prospecto, prospecto_query)

                    add_status("Prospecto generado exitosamente", "✅")

                    # Add to history
                    st.session_state.prospecto_history.append({
                        "query": prospecto_query,
                        "prospecto": response["prospecto"],
                        "context": response["context"],
                        "medication": response["medication"]
                    })

                    add_status("Guardado en historial", "💾")
                    
                    # Display the prospecto - use proper name from response
                    medication_name = response.get('medication', 'Medicamento')
                    # Ensure we show the proper name, not just registration number
                    if medication_name.isdigit() or medication_name.startswith("Nº"):
                        if 'context' in response and 'INFORMACIÓN BÁSICA DEL MEDICAMENTO:' in response['context']:
                            # Extract name from context if registration number is shown
                            name_match = re.search(r'- Nombre:\s*(.*?)(?:\n|$)', response['context'])
                            if name_match:
                                medication_name = name_match.group(1).strip()
                    
                    st.subheader(f"Prospecto para: {medication_name}")
                    
                    # Create a container and use regular markdown rendering to avoid formatting issues
                    prospecto_container = st.container()
                    with prospecto_container:
                        # Apply container styling to a div, but render the content as normal markdown
                        st.markdown('<div class="prospecto-container">', unsafe_allow_html=True)
                        st.markdown(response["prospecto"])  # Let Streamlit handle markdown rendering
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Option to download
                    prospecto_text = f"""# PROSPECTO: INFORMACIÓN PARA EL USUARIO

{response["prospecto"]}

---
Generado para: {medication_name}
Fecha de generación: {datetime.now().strftime("%d/%m/%Y")}
"""
                    st.download_button(
                        label="Descargar prospecto",
                        data=prospecto_text,
                        file_name=f"prospecto_{medication_name.replace(' ', '_')[:30]}.md",
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
                # Extract proper medication name for display
                medication_name = item.get('medication', 'Medicamento')
                if medication_name.isdigit() or medication_name.startswith("Nº"):
                    if 'context' in item and 'INFORMACIÓN BÁSICA DEL MEDICAMENTO:' in item['context']:
                        name_match = re.search(r'- Nombre:\s*(.*?)(?:\n|$)', item['context'])
                        if name_match:
                            medication_name = name_match.group(1).strip()
                
                with st.expander(f"Prospecto: {medication_name}"):
                    # Use proper markdown rendering to avoid formatting issues
                    st.markdown('<div class="prospecto-container">', unsafe_allow_html=True)
                    st.markdown(item["prospecto"])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.download_button(
                        label="Descargar",
                        data=f"""# PROSPECTO: INFORMACIÓN PARA EL USUARIO\n\n{item["prospecto"]}\n\n---\nGenerado para: {medication_name}\nFecha de generación: {datetime.now().strftime("%d/%m/%Y")}""",
                        file_name=f"prospecto_{medication_name.replace(' ', '_')[:30]}_{i}.md",
                        mime="text/markdown"
                    )
