import streamlit as st
import asyncio
import openai
from openai import AsyncOpenAI
import re
import os
from datetime import datetime  # Fixed import
from dotenv import load_dotenv
from formulacion import FormulationAgent
from perplexity_client import PerplexityClient
from prospecto import ProspectoGenerator  # New import for ProspectoGenerator
from config import Config

# Load environment variables (for local development)
load_dotenv()

# Configure page
st.set_page_config(page_title="CIMA Assistant", layout="wide")

# Professional Dashboard CSS
st.markdown("""
<style>
    /* Professional font stack */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Roboto, sans-serif !important;
    }

    /* Clean spacing */
    .main .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    .stTabs [data-baseweb="tab-panel"] {padding-top: 1.5rem;}

    /* Professional buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Info box */
    .info-box {
        background: #F0F9FF;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        margin-bottom: 20px;
        border-radius: 6px;
        font-size: 0.9em;
        color: #1E40AF;
    }

    /* Reasoning box */
    .reasoning-box {
        background: #FEFCE8;
        border-left: 4px solid #EAB308;
        padding: 16px 20px;
        margin-bottom: 20px;
        border-radius: 6px;
    }

    .reasoning-box h4 {
        font-size: 0.95em;
        color: #854D0E;
        margin-bottom: 10px;
        font-weight: 600;
    }

    /* Professional references */
    .reference-item {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.85em;
        transition: all 0.2s;
    }

    .reference-item:hover {
        border-color: #9CA3AF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .reference-title {
        font-weight: 600;
        color: #111827;
        font-size: 0.9em;
        margin-bottom: 4px;
    }

    .reference-url {
        color: #6366F1;
        word-break: break-all;
        font-size: 0.85em;
    }

    /* Activity indicator */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .activity-indicator, .thinking-indicator {
        background: #EEF2FF;
        border-left: 4px solid #6366F1;
        padding: 16px 20px;
        margin-bottom: 16px;
        border-radius: 6px;
        animation: pulse 2s ease-in-out infinite;
    }

    .activity-indicator h4, .thinking-indicator h4 {
        font-size: 0.9em;
        color: #4338CA;
        margin: 0;
    }

    /* Prospecto container */
    .prospecto-container {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 20px;
        font-family: Arial, sans-serif;
        line-height: 1.6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .prospecto-title {
        text-align: center;
        font-weight: bold;
        margin-bottom: 15px;
        font-size: 16px;
    }

    .prospecto-medication {
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
        font-size: 15px;
    }

    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Professional tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        font-weight: 500;
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

# Add function to initialize the ProspectoGenerator
@st.cache_resource
def get_prospecto_generator():
    """Get ProspectoGenerator with proper API key handling"""
    # Try to get API key from Streamlit secrets first (for cloud deployment)
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Fall back to environment variables or Config
        api_key = os.getenv("OPENAI_API_KEY") or Config.OPENAI_API_KEY
    
    if not api_key:
        st.error("No se ha encontrado la API key de OpenAI. Verifique los secretos de Streamlit, variables de entorno o el archivo config.py")
        return None
        
    return ProspectoGenerator(AsyncOpenAI(api_key=api_key))

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
perplexity_client = get_perplexity_client()
prospecto_generator = get_prospecto_generator()
    
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
        st.session_state.prospecto_history = []
        st.session_state.messages = []
        if perplexity_client:
            perplexity_client.clear_history()
        st.rerun()

# Main tabs - Add new Prospectos tab
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

# New tab for Prospectos with improved display
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
            value=st.session_state.current_prospecto_query,
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
                    
                    # Use improved container styling for AEMPS format
                    st.markdown(f"""
                    <div class="prospecto-container">
                    {clean_prospecto}
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
                    # Clean up any formatting artifacts before displaying
                    clean_prospecto = item["prospecto"]
                    clean_prospecto = clean_prospecto.replace('$', '')
                    clean_prospecto = clean_prospecto.replace('**', '')
                    clean_prospecto = clean_prospecto.replace('*', '')
                    clean_prospecto = re.sub(r'```[a-zA-Z]*', '', clean_prospecto)
                    clean_prospecto = clean_prospecto.replace('```', '')
                    
                    # Use improved container styling for AEMPS format
                    st.markdown(f"""
                    <div class="prospecto-container">
                    {clean_prospecto}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.download_button(
                        label="Descargar",
                        data=f"""PROSPECTO: INFORMACIÓN PARA EL USUARIO\n\n{clean_prospecto}\n\n---\nGenerado para: {medication_name}\nFecha de generación: {datetime.now().strftime("%d/%m/%Y")}""",
                        file_name=f"prospecto_{medication_name.replace(' ', '_')[:30]}_{i}.md",
                        mime="text/markdown"
                    )
                    
    with hist_tab3:
        st.header("Historial de consultas CIMA")
        
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