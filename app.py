import streamlit as st
import asyncio
import openai
import re
import os
from dotenv import load_dotenv

# Load environment variables
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

# Title
st.title("🧪 CIMA Assistant")
st.markdown("### *Sistema inteligente de consulta para formulación magistral y CIMA*")

# Simple page to verify app is working
st.success("El sistema está funcionando correctamente.")
st.info("Se ha iniciado en modo de diagnóstico para garantizar estabilidad. Pronto estarán disponibles todas las funcionalidades.")

# Basic tab structure 
tab1, tab2, tab3 = st.tabs(["Formulación Magistral", "Consultas CIMA", "Historial"])

with tab1:
    st.write("### Asistente para formulación magistral basado en CIMA")
    st.markdown("""
    <div class="info-box">
    Ingrese su consulta sobre formulación magistral. Especifique el principio activo, 
    concentración deseada y tipo de formulación para obtener mejores resultados.
    </div>
    """, unsafe_allow_html=True)
    
    query_fm = st.text_area(
        "Ingrese su consulta sobre formulación:",
        height=100, 
        placeholder="Ejemplo: Suspensión de Ibuprofeno 100mg/ml para uso pediátrico",
        disabled=True
    )
    
    st.button("Consultar Formulación", type="primary", disabled=True)

with tab2:
    st.write("### Chat con experto CIMA")
    st.markdown("""
    <div class="info-box">
    Realice consultas sobre medicamentos registrados en CIMA. 
    Puede preguntar sobre indicaciones, contraindicaciones, efectos adversos, etc.
    </div>
    """, unsafe_allow_html=True)
    
    st.text_area("Mensaje:", disabled=True)
    st.button("Enviar", disabled=True)

with tab3:
    st.header("Historial de formulaciones")
    st.info("El historial estará disponible próximamente.")

# Display environment check information
st.subheader("Información de diagnóstico")

try:
    # Check OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_status = "✅ API Key configurada" if openai_api_key else "❌ API Key no encontrada"
    st.write(f"OpenAI: {openai_status}")
    
    # Check event loop
    try:
        loop = asyncio.get_event_loop()
        loop_status = f"✅ Event loop disponible ({loop})"
    except RuntimeError as e:
        loop_status = f"❌ Error de event loop: {str(e)}"
    st.write(f"AsyncIO: {loop_status}")
    
    # Check Python version
    import sys
    st.write(f"Python: {sys.version}")
    
    # Check libraries
    import pkg_resources
    libraries = ['streamlit', 'openai', 'aiohttp', 'python-dotenv', 'httpx', 'nest-asyncio']
    st.write("Librerías instaladas:")
    for lib in libraries:
        try:
            version = pkg_resources.get_distribution(lib).version
            st.write(f"- {lib}: ✅ v{version}")
        except pkg_resources.DistributionNotFound:
            st.write(f"- {lib}: ❌ No instalada")

except Exception as e:
    st.error(f"Error en diagnóstico: {str(e)}")

# Contacto e instrucciones
st.markdown("""
---
### Próximos pasos
1. Verifique que todas las librerías están correctamente instaladas
2. Compruebe que la API Key de OpenAI está configurada en el archivo .env
3. Reinicie la aplicación después de corregir cualquier problema detectado

Para restaurar todas las funcionalidades, por favor contacte al soporte técnico.
""")
