import streamlit as st
import asyncio
import nest_asyncio
from openai import AsyncOpenAI
from formulacion import FormulationAgent, CIMAExpertAgent
from config import Config
import re
import os
from openai_client import create_async_openai_client

# Apply nest_asyncio to allow nested event loops
# This helps prevent "Event loop is closed" errors in Streamlit
nest_asyncio.apply()

# Configure page settings
st.set_page_config(page_title="CIMA Assistant", layout="wide")

# Ensure we have a single, reusable event loop
@st.cache_resource
def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# Get or create event loop
loop = get_event_loop()

# Initialize async resources with proper lifecycle management
@st.cache_resource
def init_agents():
    """Initialize OpenAI client and agents with proper async resource management"""
    openai_client = create_async_openai_client(api_key=Config.OPENAI_API_KEY)
    
    # Initialize the agents with the client
    formulation_agent = FormulationAgent(openai_client)
    cima_expert_agent = CIMAExpertAgent(openai_client)
    
    return formulation_agent, cima_expert_agent

# Run async function within our managed event loop
def run_async(coro):
    """Run an async function within our managed event loop"""
    return loop.run_until_complete(coro)

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
                    
                    # Get response using our managed event loop
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
                    
                    # Process response using our managed event loop
                    response = run_async(st.session_state.agents[1].chat(prompt))
                    
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
                    st.markdown(f"Error: {str(e)}")
                    
        # Add to session state
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    
    # Button for new conversation
    if st.button("Nueva conversación", key="new_chat"):
        st.session_state.messages = []
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