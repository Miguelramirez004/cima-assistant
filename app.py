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

# Custom CSS styling
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
if 'show_reasoning' not in st.session_state:
    st.session_state.show_reasoning = True

# Silently initialize clients without showing status messages
openai_client = get_openai_client()
perplexity_client = get_perplexity_client()
    
# Title
st.title("🧪 CIMA Assistant")
st.markdown("### *Sistema inteligente de consulta para formulación magistral y CIMA*")

# Sidebar
with st.sidebar:
    st.header("Información")
    st.markdown("""
    Este asistente utiliza:
    
    - Base de datos CIMA para formulaciones magistrales
    - Consultas sobre medicamentos
    - Referencias a fuentes oficiales
    """)
    
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

        # Process and display assistant response
        with st.chat_message("assistant"):
            # Create a placeholder for the "thinking" animation
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("""
            <div class="thinking-indicator">
            <h4>💭 Pensando...</h4>
            <p>Estoy analizando la información médica disponible sobre su consulta. Este proceso puede tomar unos segundos...</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Consultando base de conocimiento médico..."):
                try:
                    # Get Perplexity client
                    perplexity_client = get_perplexity_client()
                    if not perplexity_client:
                        st.error("No se puede conectar con Perplexity. Verifique su API key.")
                    else:
                        # Process the request (fallback to sync method if async fails)
                        try:
                            response = run_async(perplexity_client.ask_cima_question_async, prompt)
                        except Exception as async_err:
                            # Fall back to sync method if async fails
                            response = perplexity_client.ask_cima_question(prompt)
                        
                        # Clear the thinking animation
                        thinking_placeholder.empty()
                        
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
                    error_message = f"Error: {str(e)}"
                    st.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    
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
