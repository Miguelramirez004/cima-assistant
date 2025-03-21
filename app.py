import streamlit as st
import asyncio
import openai
from openai import AsyncOpenAI
import re
import os
from dotenv import load_dotenv
from formulacion import FormulationAgent
from perplexity_client import PerplexityClient
from config import Config

# Load environment variables (for local development)
load_dotenv()

# Configure page
st.set_page_config(page_title="CIMA Assistant", layout="wide")

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

# Global Perplexity client for CIMA consultations
@st.cache_resource
def get_perplexity_client():
    """Get Perplexity client with proper API key handling"""
    # Try to get API key from Streamlit secrets first (for cloud deployment)
    try:
        api_key = st.secrets["PERPLEXITY_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Fall back to environment variables or Config
        api_key = os.getenv("PERPLEXITY_API_KEY") or Config.PERPLEXITY_API_KEY
    
    if not api_key:
        st.error("No se ha encontrado la API key de Perplexity. Verifique los secretos de Streamlit, variables de entorno o el archivo config.py")
        return None
        
    return PerplexityClient(api_key=api_key)

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
if 'use_langgraph' not in st.session_state:
    st.session_state.use_langgraph = True

# Check API keys at startup
openai_client = get_openai_client()
perplexity_client = get_perplexity_client()

if openai_client:
    st.success("✅ Conexión a OpenAI configurada correctamente")
else:
    st.error("❌ Error: No se pudo establecer conexión con OpenAI. Por favor configure la API key en los secretos de Streamlit.")

if perplexity_client:
    st.success("✅ Conexión a Perplexity configurada correctamente")
else:
    st.error("❌ Error: No se pudo establecer conexión con Perplexity. Por favor configure la API key en los secretos de Streamlit.")
    
# Title
st.title("🧪 CIMA Assistant")
st.markdown("### *Sistema inteligente de consulta para formulación magistral y CIMA*")

# Sidebar
with st.sidebar:
    st.header("Información")
    st.markdown("""
    Este asistente utiliza:
    
    - Base de datos CIMA para formulaciones magistrales
    - Perplexity AI (Sonar Pro) para consultas sobre medicamentos
    - Referencias a fuentes oficiales
    """)
    
    # Add search mode setting for formulation
    st.header("Ajustes")
    use_langgraph = st.toggle("Usar búsqueda avanzada para formulación", value=st.session_state.use_langgraph)
    
    # Update session state if toggle changed
    if use_langgraph != st.session_state.use_langgraph:
        st.session_state.use_langgraph = use_langgraph
        st.info(f"Modo de búsqueda para formulación: {'Avanzado' if use_langgraph else 'Estándar'}")
    
    # Explain Sonar Pro
    with st.expander("Consultas CIMA - Tecnología"):
        st.markdown("""
        Para las consultas de medicamentos, esta aplicación utiliza el modelo Sonar Pro de Perplexity AI, 
        que proporciona:
        
        - Respuestas detalladas basadas en conocimiento médico actualizado
        - Mayor precisión en la información
        - Capacidad avanzada de razonamiento para responder consultas complejas
        
        A diferencia de la implementación anterior, este modelo proporciona información 
        más actualizada y contextualizada.
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
        st.session_state.messages = []
        if perplexity_client:
            perplexity_client.clear_history()
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
    st.write("### Chat con experto CIMA (Perplexity Sonar Pro)")
    st.markdown("""
    <div class="info-box">
    Realice consultas sobre medicamentos utilizando el modelo Sonar Pro de Perplexity AI.
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
            
            with st.spinner("Consultando base de conocimiento médico..."):
                try:
                    # Show progress updates
                    with progress_placeholder.container():
                        progress_bar = st.progress(0)
                    
                    status_text.text("Procesando su consulta...")
                    progress_bar.progress(30)
                    
                    # Get Perplexity client
                    perplexity_client = get_perplexity_client()
                    if not perplexity_client:
                        st.error("No se puede conectar con Perplexity. Verifique su API key.")
                    else:
                        # Process the request (fallback to sync method if async fails)
                        try:
                            response = run_async(perplexity_client.ask_cima_question_async, prompt)
                        except Exception as async_err:
                            st.warning(f"Modo asíncrono no disponible: {str(async_err)}", icon="⚠️")
                            response = perplexity_client.ask_cima_question(prompt)
                        
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
                        with st.expander("Información sobre la fuente"):
                            st.markdown(response["context"])
                            st.markdown("Esta respuesta ha sido generada utilizando el modelo Perplexity Sonar Pro, "
                                   "que integra conocimiento médico actualizado con capacidades de razonamiento avanzadas.")
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    raise
                    
        # Add to session state (only if successful)
        if "answer" in response and response["success"]:
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    
    # Button for new conversation
    if st.button("Nueva conversación", key="new_chat"):
        st.session_state.messages = []
        if perplexity_client:
            perplexity_client.clear_history()
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
