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

# Minimalist CSS styling
st.markdown("""
<style>
    /* Clean, professional font stack */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
    }

    /* Cleaner spacing */
    .main .block-container {padding-top: 1.5rem;}
    .stTabs [data-baseweb="tab-panel"] {padding-top: 1rem;}

    /* Subtle, professional buttons */
    div.stButton > button:first-child {
        background-color: #4A5568;
        color: white;
        border: none;
        border-radius: 4px;
    }
    div.stButton > button:hover {
        background-color: #2D3748;
    }

    /* Minimal info box - subtle border only */
    .info-box {
        border-left: 3px solid #CBD5E0;
        padding: 12px 16px;
        margin-bottom: 16px;
        background-color: transparent;
        font-size: 0.9em;
        color: #4A5568;
    }

    /* Clean reasoning box */
    .reasoning-box {
        background-color: #F7FAFC;
        border-left: 3px solid #4A5568;
        padding: 12px 16px;
        margin-bottom: 16px;
        border-radius: 4px;
    }

    .reasoning-box h4 {
        font-size: 0.9em;
        color: #4A5568;
        margin-bottom: 8px;
    }

    /* Compact references */
    .reference-item {
        background-color: transparent;
        border-left: 2px solid #E2E8F0;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.85em;
    }

    .reference-title {
        font-weight: 600;
        color: #2D3748;
        font-size: 0.9em;
    }

    .reference-url {
        color: #4A5568;
        word-break: break-all;
        font-size: 0.85em;
    }

    /* Minimal thinking indicator */
    @keyframes thinking-animation {
        0% { opacity: 0.5; }
        50% { opacity: 1.0; }
        100% { opacity: 0.5; }
    }

    .thinking-indicator {
        animation: thinking-animation 1.5s infinite;
        background-color: #F7FAFC;
        border-left: 3px solid #A0AEC0;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 4px;
    }

    .thinking-indicator h4 {
        font-size: 0.9em;
        color: #4A5568;
    }

    /* Clean prospecto container */
    .prospecto-container {
        background-color: transparent;
        border-left: 3px solid #CBD5E0;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 16px;
    }

    /* Minimal debug info */
    .debug-info {
        background-color: #F7FAFC;
        border: 1px solid #E2E8F0;
        padding: 8px;
        margin-top: 8px;
        font-size: 0.75em;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
        white-space: pre-wrap;
        overflow-x: auto;
        border-radius: 4px;
    }

    /* Better chat message spacing */
    [data-testid="stChatMessage"] {
        margin-bottom: 0.5rem;
    }

    /* Hide unnecessary elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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
    st.session_state.current_prospecto_query = ""  # Add dedicated variable for prospecto queries
    st.session_state.use_langgraph = True  # Default to using the improved search
    st.session_state.show_reasoning = True  # New setting for showing reasoning
    st.session_state.debug_mode = False    # Debug mode

# Clean title
st.title("🧪 CIMA Assistant")
st.caption("Sistema de consulta para formulación magistral y medicamentos")

# Minimal Sidebar
with st.sidebar:
    st.markdown("### Ajustes")

    # Compact settings toggles
    use_langgraph = st.toggle("Búsqueda avanzada", value=st.session_state.use_langgraph,
                               help="Activa búsqueda avanzada para formulaciones")

    if use_langgraph != st.session_state.use_langgraph:
        st.session_state.use_langgraph = use_langgraph
        if st.session_state.resources and st.session_state.resources[0]:
            st.session_state.resources[0].use_langgraph = use_langgraph

    show_reasoning = st.toggle("Mostrar razonamiento", value=st.session_state.show_reasoning,
                                help="Muestra el proceso de razonamiento del asistente")
    if show_reasoning != st.session_state.show_reasoning:
        st.session_state.show_reasoning = show_reasoning

    # Debug mode (hidden unless query param is set)
    debug_param = st.query_params.get("debug")
    if debug_param == "true":
        debug_mode = st.toggle("Depuración", value=st.session_state.debug_mode)
        if debug_mode != st.session_state.debug_mode:
            st.session_state.debug_mode = debug_mode

    st.divider()

    # Compact history display
    if st.session_state.search_history:
        with st.expander("📝 Búsquedas recientes"):
            for query in list(st.session_state.search_history)[-5:]:
                st.caption(query)

    # Compact action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Limpiar", use_container_width=True):
            st.session_state.search_history = set()
            st.session_state.formulation_history = []
            st.session_state.prospecto_history = []
            if st.session_state.resources and st.session_state.resources[1]:
                st.session_state.resources[1].clear_history()
            st.session_state.messages = []
            st.rerun()

    with col2:
        # Diagnostic tool in expander (less prominent)
        if st.button("Diagnóstico", use_container_width=True, key="diagnostico_btn"):
            st.session_state.show_diagnostico = not st.session_state.get('show_diagnostico', False)

    # Show diagnostic in expander when button is clicked
    if st.session_state.get('show_diagnostico', False):
        with st.expander("🔧 Diagnóstico CIMA", expanded=True):
            import requests

            try:
                # Test 1: Basic connection
                response = requests.get("https://cima.aemps.es/cima/rest/medicamentos", timeout=5)
                if response.status_code == 200:
                    st.success("✅ Conexión CIMA OK")
                else:
                    st.error(f"❌ Error: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)[:50]}")

            try:
                # Test 2: Medication access
                nregistro = "78929"
                response = requests.get(f"https://cima.aemps.es/cima/rest/medicamento?nregistro={nregistro}", timeout=5)
                if response.status_code == 200:
                    st.success("✅ Acceso medicamentos OK")
                else:
                    st.error(f"❌ Error medicamentos: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Error medicamentos: {str(e)[:50]}")

# Main tabs - Updated to include Prospectos tab
tab1, tab2, tab3, tab4 = st.tabs(["Formulación Magistral", "Consultas CIMA", "Prospectos", "Historial"])

with tab1:
    st.markdown("### Formulación Magistral")

    # Minimal info hint
    st.markdown("""
    <div class="info-box">
    Especifique el principio activo, concentración y tipo de formulación.
    </div>
    """, unsafe_allow_html=True)

    # Handle query text area
    query_fm = st.text_area(
        "Consulta:",
        value=st.session_state.current_query,
        height=100,
        placeholder="Ej: Suspensión de Ibuprofeno 100mg/ml para uso pediátrico"
    )

    # Collapsed examples
    with st.expander("💡 Ver ejemplos"):
        example_queries = [
            "Suspensión de Omeprazol 2mg/ml",
            "Crema de Hidrocortisona al 1%",
            "Cápsulas de Melatonina 3mg",
            "Gel de Metronidazol 0.75%",
            "Solución de Minoxidil 5%",
            "MINOXIDIL BIORGA"
        ]

        cols = st.columns(2)
        for i, example in enumerate(example_queries):
            with cols[i % 2]:
                if st.button(example, key=f"fm_ex_{i}", use_container_width=True):
                    st.session_state.current_query = example if example != "MINOXIDIL BIORGA" else "Encontrar información sobre MINOXIDIL BIORGA"
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

            # Simple progress indicator
            with st.spinner("Procesando consulta..."):
                try:
                    # Set the agent's search mode based on current setting
                    st.session_state.resources[0].use_langgraph = st.session_state.use_langgraph

                    # Process response using our managed event loop
                    response = run_async(st.session_state.resources[0].answer_question(query_fm))

                    # Check if it's a prospecto request that was redirected
                    if "use la pestaña 'Prospectos'" in response.get("answer", ""):
                        st.info("Esta consulta es para generar un prospecto. Use la pestaña 'Prospectos'.")
                        st.session_state.current_prospecto_query = query_fm
                        st.rerun()

                    # Store in formulation history
                    st.session_state.formulation_history.append({
                        "query": query_fm,
                        "response": response["answer"],
                        "context": response["context"],
                        "references": response["references"]
                    })
                    
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
    st.markdown("### Consultas CIMA")

    # Minimal info hint
    st.markdown("""
    <div class="info-box">
    Pregunte sobre indicaciones, dosis, contraindicaciones, efectos secundarios, etc.
    </div>
    """, unsafe_allow_html=True)

    # Example section - more compact
    with st.expander("💡 Ejemplos"):
        st.caption("¿Cuáles son los efectos secundarios del ibuprofeno?")
        st.caption("¿Qué dosis de paracetamol es segura para niños?")
        st.caption("¿Qué interacciones tiene la simvastatina?")
        st.caption("¿Es seguro tomar metformina durante el embarazo?")
    
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
            # Minimal thinking indicator
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("""
            <div class="thinking-indicator">
            <h4>💭 Analizando...</h4>
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

# Minimalist Prospectos tab
with tab3:
    st.markdown("### Prospectos")

    # Minimal info hint
    st.markdown("""
    <div class="info-box">
    Genere prospectos según normativa AEMPS. Especifique nombre o principio activo.
    </div>
    """, unsafe_allow_html=True)

    # Input for prospecto query
    prospecto_query = st.text_area(
        "Consulta:",
        value=st.session_state.current_prospecto_query,
        height=100,
        placeholder="Ej: Generar prospecto para Ibuprofeno 600mg"
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
                    st.rerun()
    
    # Generate button
    if st.button("Generar Prospecto", type="primary", key="generate_prospecto"):
        if not prospecto_query:
            st.warning("Por favor ingrese una consulta")
        else:
            # Add to search history
            st.session_state.search_history.add(prospecto_query)

            # Simple progress indicator
            with st.spinner("Generando prospecto..."):
                try:
                    # Get prospecto generator from resources
                    prospecto_generator = st.session_state.resources[2]

                    # Generate prospecto
                    response = run_async(prospecto_generator.generate_prospecto, prospecto_query)

                    # Add to history
                    st.session_state.prospecto_history.append({
                        "query": prospecto_query,
                        "prospecto": response["prospecto"],
                        "context": response["context"],
                        "medication": response["medication"]
                    })
                    
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
