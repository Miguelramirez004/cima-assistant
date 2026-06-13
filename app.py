import streamlit as st
import asyncio
import openai
from openai import AsyncOpenAI
import re
import os
from datetime import datetime  # Fixed import
from dotenv import load_dotenv
from formulacion import FormulationAgent
from cima_rag import CIMARagAgent
from prospecto import ProspectoGenerator  # New import for ProspectoGenerator
from config import Config
from security import escape_html, safe_url

# Load environment variables (for local development)
load_dotenv()

# Configure page
st.set_page_config(
    page_title="CIMA Assistant",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern minimalist design system
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --accent: #0D9488;
        --accent-soft: #F0FDFA;
        --ink: #1E293B;
        --ink-soft: #64748B;
        --ink-faint: #94A3B8;
        --line: #E2E8F0;
        --surface: #F8FAFC;
        --radius: 10px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        color: var(--ink);
        -webkit-font-smoothing: antialiased;
    }

    /* Layout */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    .stTabs [data-baseweb="tab-panel"] {padding-top: 1.5rem;}

    /* Hide Streamlit chrome */
    #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden; height: 0;}

    /* App header */
    .app-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding-bottom: 1.25rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid var(--line);
    }
    .app-logo {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: var(--accent);
        color: #fff;
        font-size: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .app-title {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.85rem;
        color: var(--ink-soft);
        margin: 2px 0 0 0;
        font-weight: 400;
    }

    /* Section headings */
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        margin: 0 0 0.25rem 0;
    }
    .section-caption {
        font-size: 0.85rem;
        color: var(--ink-soft);
        margin: 0 0 1rem 0;
        line-height: 1.5;
    }

    /* Tabs: minimal underline style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid var(--line);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        font-weight: 500;
        font-size: 0.9rem;
        color: var(--ink-soft);
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--accent);
        height: 2px;
    }
    .stTabs [data-baseweb="tab-border"] {display: none;}

    /* Buttons */
    div.stButton > button {
        background: #FFFFFF;
        color: var(--ink);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 0.45rem 1.1rem;
        font-weight: 500;
        font-size: 0.85rem;
        transition: border-color 0.15s ease, background 0.15s ease;
        box-shadow: none;
    }
    div.stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
        background: var(--accent-soft);
    }
    div.stButton > button[kind="primary"] {
        background: var(--accent);
        color: #FFFFFF;
        border: 1px solid var(--accent);
        font-weight: 600;
        padding: 0.55rem 1.5rem;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #0F766E;
        border-color: #0F766E;
        color: #FFFFFF;
    }

    /* Download buttons */
    div.stDownloadButton > button {
        background: #FFFFFF;
        color: var(--ink);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        font-weight: 500;
        font-size: 0.85rem;
    }
    div.stDownloadButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
    }

    /* Inputs */
    .stTextArea textarea, .stTextInput input {
        border: 1px solid var(--line) !important;
        border-radius: var(--radius) !important;
        font-size: 0.9rem;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.12) !important;
    }

    /* Info box */
    .info-box {
        background: var(--surface);
        border: 1px solid var(--line);
        padding: 12px 16px;
        margin: 0.5rem 0 1rem 0;
        border-radius: var(--radius);
        font-size: 0.85rem;
        color: var(--ink-soft);
        line-height: 1.5;
    }

    /* Prospecto container */
    .prospecto-container {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 28px 32px;
        margin-bottom: 20px;
        line-height: 1.65;
        font-size: 0.9rem;
    }
    .prospecto-title {
        text-align: center;
        font-weight: 700;
        margin-bottom: 14px;
        font-size: 1rem;
        letter-spacing: -0.01em;
    }
    .prospecto-medication {
        text-align: center;
        font-weight: 600;
        margin-bottom: 20px;
        font-size: 0.95rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] .block-container {padding-top: 1.5rem;}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--ink-soft);
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {
        font-size: 0.84rem;
        color: var(--ink-soft);
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: var(--radius);
    }
    div[data-testid="stExpander"] summary {
        font-size: 0.88rem;
        font-weight: 500;
    }

    /* Chat — clean light style (Claude-like). Rendered with our own markup
       (not st.chat_message) so it does not depend on Streamlit internal DOM. */
    .chat-row {
        display: flex;
        margin: 0.35rem 0;
    }
    .chat-row.user { justify-content: flex-end; }
    .chat-row.assistant { justify-content: flex-start; }
    .user-bubble {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 18px 18px 4px 18px;
        padding: 10px 16px;
        max-width: 75%;
        font-size: 0.92rem;
        color: var(--ink);
        line-height: 1.5;
        word-wrap: break-word;
    }
    .stChatInput textarea {
        border-radius: 14px !important;
        font-size: 0.9rem !important;
    }
    .stChatInput {
        border-top: 1px solid var(--line);
        padding-top: 0.75rem;
    }

    /* Retrieval trace (graph steps) */
    .trace-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
        padding: 4px 0;
    }
    .trace-step {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        font-size: 0.83rem;
        color: var(--ink-soft);
        line-height: 1.45;
    }
    .trace-num {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.7rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 1px;
    }

    /* Source pills */
    .source-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1px solid var(--line);
    }
    .source-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--ink-faint);
        font-weight: 600;
        margin-right: 2px;
    }
    a.source-pill, span.source-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--accent-soft);
        color: var(--accent);
        border: 1px solid #CCFBF1;
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 500;
        text-decoration: none;
        transition: background 0.15s ease, border-color 0.15s ease;
        max-width: 320px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    a.source-pill:hover {
        background: #CCFBF1;
        border-color: var(--accent);
        text-decoration: none;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: var(--accent);
    }

    /* Alerts: flatter look */
    div[data-testid="stAlert"] {
        border-radius: var(--radius);
        border: 1px solid var(--line);
        font-size: 0.87rem;
    }
</style>
""", unsafe_allow_html=True)

# Simple, reliable async helper function for Streamlit Cloud
def run_async(async_func, *args, **kwargs):
    """Run an async function in a way compatible with Streamlit Cloud"""
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(async_func(*args, **kwargs))
    except Exception as e:
        st.error(f"Async error: {str(e)}")
        raise
    finally:
        # Don't close the loop to avoid issues with Streamlit's event loop
        pass

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

# Global CIMA RAG agent for CIMA consultations (replaces Perplexity)
@st.cache_resource
def get_cima_rag_agent():
    """Get CIMA RAG agent backed by the official AEMPS REST API"""
    openai_client = get_openai_client()
    if not openai_client:
        return None
    return CIMARagAgent(openai_client)

# Add function to initialize the ProspectoGenerator
def get_prospecto_generator():
    """Get a ProspectoGenerator. NOT cached on purpose: a fresh instance per
    script run gets a fresh aiohttp session bound to the current event loop,
    avoiding the cross-loop reuse that breaks cached async clients in Streamlit.
    The underlying OpenAI client is the cached one."""
    openai_client = get_openai_client()
    if not openai_client:
        return None
    return ProspectoGenerator(openai_client)

# Initialize session state variables if not already present
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'formulation_history' not in st.session_state:
    st.session_state.formulation_history = []
if 'prospecto_history' not in st.session_state:
    st.session_state.prospecto_history = []
if 'search_history' not in st.session_state:
    st.session_state.search_history = set()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'current_prospecto_query' not in st.session_state:
    st.session_state.current_prospecto_query = ""
if 'use_langgraph' not in st.session_state:
    st.session_state.use_langgraph = True
if 'show_reasoning' not in st.session_state:
    st.session_state.show_reasoning = True

# Silently initialize clients without showing status messages
openai_client = get_openai_client()
cima_rag_agent = get_cima_rag_agent()
prospecto_generator = get_prospecto_generator()
    
# Header
st.markdown("""
<div class="app-header">
    <div class="app-logo">⚕</div>
    <div>
        <p class="app-title">CIMA Assistant</p>
        <p class="app-subtitle">Consulta inteligente para formulación magistral y medicamentos</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Información")
    st.markdown("""
    Este asistente utiliza:

    - Base de datos CIMA para formulaciones magistrales
    - Consultas sobre medicamentos
    - Referencias a fuentes oficiales
    """)

    st.divider()

    # Add search mode setting for formulation
    st.header("Ajustes")
    use_langgraph = st.toggle("Usar búsqueda avanzada para formulación", value=st.session_state.use_langgraph)
    
    # Update session state if toggle changed
    if use_langgraph != st.session_state.use_langgraph:
        st.session_state.use_langgraph = use_langgraph
        st.info(f"Modo de búsqueda para formulación: {'Avanzado' if use_langgraph else 'Estándar'}")
    
    # Add toggle for showing reasoning process
    show_reasoning = st.toggle("Mostrar proceso de razonamiento", value=st.session_state.show_reasoning)
    if show_reasoning != st.session_state.show_reasoning:
        st.session_state.show_reasoning = show_reasoning
        st.info(f"Visualización de razonamiento: {'Activado' if show_reasoning else 'Desactivado'}")

    st.divider()

    st.header("Historial de búsquedas")
    if st.session_state.search_history:
        for query in list(st.session_state.search_history)[-5:]:
            st.markdown(f"- {query}")
    else:
        st.markdown("No hay búsquedas recientes")
    
    if st.button("Limpiar historial"):
        st.session_state.search_history = set()
        st.session_state.formulation_history = []
        st.session_state.prospecto_history = []
        st.session_state.messages = []
        if cima_rag_agent:
            cima_rag_agent.clear_history()
        st.rerun()

# Main tabs - Add new Prospectos tab
tab1, tab2, tab3, tab4 = st.tabs(["Formulación Magistral", "Consultas CIMA", "Prospectos", "Historial"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <p class="section-title">Formulación magistral</p>
        <p class="section-caption">Especifique el principio activo, la concentración deseada
        y el tipo de formulación para obtener mejores resultados.</p>
        """, unsafe_allow_html=True)
        
        # Handle query text area
        query_fm = st.text_area(
            "Ingrese su consulta sobre formulación:",
            value=st.session_state.current_query,
            height=100, 
            placeholder="Ejemplo: Suspensión de Ibuprofeno 100mg/ml para uso pediátrico"
        )
    
    with col2:
        st.markdown('<p class="section-title">Ejemplos</p>', unsafe_allow_html=True)
        example_queries = [
                "Suspensión de Ibuprofeno 100mg/ml pediátrico",
                "Cápsulas de Omeprazol 20mg",
                "Comprimidos de Simvastatina 20mg",
                "Solución MINOXIDIL BIORGA 50mg/ml",
                "Gel de Metronidazol 0.75%",
                "Crema de Hidrocortisona al 1%",
                "Comprimidos de Enalapril 10mg",
                "Suspensión de Amoxicilina 250mg/5ml",
                "Pomada de Mupirocina 2%",
                "Comprimidos de Metformina 850mg"
        ]
        
        for example in example_queries:
            if st.button(example, use_container_width=True):
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
                    
                    # Create agent for this specific request
                    openai_client = get_openai_client()
                    if not openai_client:
                        st.error("No se puede conectar con OpenAI. Verifique su API key.")
                    else:
                        formulation_agent = FormulationAgent(openai_client)
                        # Set search mode based on toggle
                        formulation_agent.use_langgraph = st.session_state.use_langgraph
                        
                        # Get response using our helper function
                        response = run_async(formulation_agent.answer_question, query_fm)
                        
                        # Check if it's a prospecto request that was redirected
                        if "use la pestaña 'Prospectos'" in response.get("answer", ""):
                            status_text.text("Redirigiendo a la sección de Prospectos...")
                            progress_bar.progress(100)
                            st.info("Esta consulta parece ser para generar un prospecto. Por favor, utilice la pestaña 'Prospectos' para esta funcionalidad.")
                            # Auto-select the Prospectos tab
                            st.session_state.current_prospecto_query = query_fm  # Preserve query for prospectos tab
                            tab3.checkbox("Redireccionar", value=True, key="redirect_to_prospectos")
                        else:
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
                        
                        # Clean up resources
                        run_async(formulation_agent.close)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab2:
    st.markdown("""
    <p class="section-title">Chat con experto CIMA</p>
    <p class="section-caption">Realice consultas sobre medicamentos: indicaciones,
    contraindicaciones, dosis, efectos secundarios y más.</p>
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
    
    def render_user_message(text: str):
        """Render the user turn as a compact right-aligned bubble (sanitized)."""
        st.markdown(
            f'<div class="chat-row user"><div class="user-bubble">{escape_html(text)}</div></div>',
            unsafe_allow_html=True,
        )

    def render_assistant_message(answer: str, reasoning: str, references: list):
        """Render an assistant chat message: collapsible retrieval trace,
        answer body and source pills. All dynamic content is sanitized."""
        # Retrieval trace as numbered steps inside a collapsed expander
        if st.session_state.show_reasoning and reasoning:
            steps = [s.lstrip("• ").strip() for s in reasoning.split("\n") if s.strip()]
            with st.expander(f"Proceso de recuperación · {len(steps)} pasos"):
                steps_html = "".join(
                    f'<div class="trace-step"><span class="trace-num">{i}</span>'
                    f'<span>{escape_html(step)}</span></div>'
                    for i, step in enumerate(steps, 1)
                )
                st.markdown(f'<div class="trace-list">{steps_html}</div>', unsafe_allow_html=True)

        # Main answer
        st.markdown(answer)

        # Sources as compact linked pills
        if references:
            pills = []
            for ref in references:
                title = escape_html(ref.get("title", ""))
                raw_url = ref.get("url", "")
                if raw_url:
                    url = safe_url(raw_url)
                    pills.append(
                        f'<a class="source-pill" href="{url}" target="_blank" '
                        f'rel="noopener noreferrer" title="{title}">📄 {title}</a>'
                    )
                else:
                    pills.append(f'<span class="source-pill">📄 {title}</span>')
            st.markdown(
                '<div class="source-row"><span class="source-label">Fuentes CIMA</span>'
                + "".join(pills) + "</div>",
                unsafe_allow_html=True,
            )

    # Chat container
    chat_container = st.container()

    # Display chat messages (custom markup, not st.chat_message)
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                render_user_message(message["content"])
            elif "reasoning" in message and "references" in message:
                render_assistant_message(
                    message["content"], message["reasoning"], message["references"]
                )
            else:
                # Plain assistant message (e.g. an error)
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Escriba su consulta sobre medicamentos..."):
        # Add to search history
        st.session_state.search_history.add(prompt)

        # Display user message
        render_user_message(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process and display assistant response
        with st.container():
            try:
                # Get the CIMA RAG agent
                cima_rag_agent = get_cima_rag_agent()
                if not cima_rag_agent:
                    st.error("No se puede conectar con OpenAI. Verifique su API key.")
                else:
                    # Run the RAG graph over the official CIMA REST API,
                    # with a live status indicator instead of a static spinner
                    with st.status("Consultando CIMA (AEMPS)...", expanded=False) as status:
                        response = run_async(cima_rag_agent.ask, prompt)
                        n_sources = len(response.get("references", []))
                        status.update(
                            label=f"Consulta completada · {n_sources} fuente(s) oficial(es)",
                            state="complete",
                        )

                    # Extract structured data from response
                    reasoning = response.get("reasoning", "")
                    answer = response.get("answer", "")
                    references = response.get("references", [])

                    render_assistant_message(answer, reasoning, references)

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
                error_message = f"Error: {str(e)}"
                st.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                    
    # Button for new conversation
    if st.button("Nueva conversación", key="new_chat"):
        st.session_state.messages = []
        if cima_rag_agent:
            cima_rag_agent.clear_history()
        st.rerun()

# New tab for Prospectos with improved display
with tab3:
    st.markdown("""
    <p class="section-title">Generador de prospectos</p>
    <p class="section-caption">Genere prospectos completos según la normativa de la AEMPS.
    Especifique el nombre del medicamento o principio activo.</p>
    """, unsafe_allow_html=True)
    
    # Create columns for input and examples
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Input for prospecto query
        prospecto_query = st.text_area(
            "Solicitud para generar un prospecto:",
            value=st.session_state.current_prospecto_query,
            height=100,
            placeholder="Ejemplo: Generar prospecto para Ibuprofeno 600mg"
        )
    
    with col2:
        st.markdown('<p class="section-title">Ejemplos</p>', unsafe_allow_html=True)
        example_queries = [
            "Generar prospecto para Ibuprofeno 600mg",
            "Prospecto de Omeprazol 20mg",
            "Crear prospecto para Amoxicilina 500mg",
            "Prospecto para MINOXIDIL BIORGA"
        ]
        
        for example in example_queries:
            if st.button(example, key=f"prospecto_{example}", use_container_width=True):
                # Store in session state to preserve across reruns
                st.session_state.current_prospecto_query = example
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
                    
                    # Clean up any formatting artifacts in the prospecto
                    clean_prospecto = response["prospecto"]
                    clean_prospecto = clean_prospecto.replace('$', '')
                    clean_prospecto = clean_prospecto.replace('**', '')
                    clean_prospecto = clean_prospecto.replace('*', '')
                    clean_prospecto = re.sub(r'```[a-zA-Z]*', '', clean_prospecto)
                    clean_prospecto = clean_prospecto.replace('```', '')
                    
                    # Use improved container styling for AEMPS format.
                    # SECURITY: el prospecto lo genera el LLM y puede reflejar la
                    # consulta del usuario o contenido de CIMA; se escapa antes de
                    # llegar al sink unsafe_allow_html para evitar XSS almacenado.
                    st.markdown(f"""
                    <div class="prospecto-container">
                    {escape_html(clean_prospecto)}
                    </div>
                    """, unsafe_allow_html=True)

                    # Option to download
                    prospecto_text = f"""PROSPECTO: INFORMACIÓN PARA EL USUARIO

{clean_prospecto}

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

# Update the Historial tab (now tab4 instead of tab3)
with tab4:
    # Create subtabs for different history types
    hist_tab1, hist_tab2, hist_tab3 = st.tabs(["Formulaciones", "Prospectos", "Consultas CIMA"])
    
    with hist_tab1:
        st.markdown('<p class="section-title">Historial de formulaciones</p>', unsafe_allow_html=True)
        
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
        st.markdown('<p class="section-title">Historial de prospectos</p>', unsafe_allow_html=True)
        
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
                    # Clean up any formatting artifacts before displaying
                    clean_prospecto = item["prospecto"]
                    clean_prospecto = clean_prospecto.replace('$', '')
                    clean_prospecto = clean_prospecto.replace('**', '')
                    clean_prospecto = clean_prospecto.replace('*', '')
                    clean_prospecto = re.sub(r'```[a-zA-Z]*', '', clean_prospecto)
                    clean_prospecto = clean_prospecto.replace('```', '')
                    
                    # Use improved container styling for AEMPS format.
                    # SECURITY: escapar el contenido generado antes del sink HTML.
                    st.markdown(f"""
                    <div class="prospecto-container">
                    {escape_html(clean_prospecto)}
                    </div>
                    """, unsafe_allow_html=True)

                    st.download_button(
                        label="Descargar",
                        data=f"""PROSPECTO: INFORMACIÓN PARA EL USUARIO\n\n{clean_prospecto}\n\n---\nGenerado para: {medication_name}\nFecha de generación: {datetime.now().strftime("%d/%m/%Y")}""",
                        file_name=f"prospecto_{medication_name.replace(' ', '_')[:30]}_{i}.md",
                        mime="text/markdown"
                    )
                    
    with hist_tab3:
        st.markdown('<p class="section-title">Historial de consultas CIMA</p>', unsafe_allow_html=True)
        
        if not st.session_state.messages:
            st.info("No hay consultas CIMA en el historial")
        else:
            # Group messages by conversation
            conversation_count = 0
            current_conversation = []
            
            for msg in st.session_state.messages:
                current_conversation.append(msg)
                
                # If this is an assistant message, it completes a Q&A pair
                if msg["role"] == "assistant":
                    conversation_count += 1
                    with st.expander(f"Consulta {conversation_count}"):
                        for conv_msg in current_conversation:
                            st.write(f"**{conv_msg['role'].capitalize()}:** {conv_msg['content'][:100]}...")
                    
                    current_conversation = []