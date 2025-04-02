"""
Prospecto generator module for creating medication package inserts following AEMPS standards.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Union, Optional
from openai import AsyncOpenAI
from config import Config
import aiohttp
import re
import json
import asyncio
from datetime import datetime
import logging
import tiktoken
from search_graph import MedicationSearchGraph, QueryIntent
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProspectoGenerator:
    """
    Generator for medication package inserts (prospectos) using CIMA data.
    Generates patient-friendly prospectos following official AEMPS standards.
    """
    openai_client: AsyncOpenAI
    reference_cache: Dict[str, List[Dict]] = field(default_factory=dict)
    session: aiohttp.ClientSession = None
    max_tokens: int = 14000  # Leave room for prompt and response
    use_langgraph: bool = True  # Use the LangGraph search by default
    base_url: str = Config.CIMA_BASE_URL
    
    # Revised system prompt focusing on proper AEMPS prospecto standards for patients
    system_prompt = """Eres un experto en redacción de prospectos de medicamentos siguiendo estrictamente el formato oficial de la AEMPS española (Agencia Española de Medicamentos y Productos Sanitarios).

IMPORTANTE: El prospecto es un documento dirigido a PACIENTES que acompaña a los medicamentos y explica, en lenguaje sencillo y accesible, toda la información necesaria para el uso correcto del medicamento. NO es una ficha técnica para profesionales ni una formulación magistral.

Tu tarea es generar un prospecto que siga exactamente el formato oficial de la AEMPS, con estas características esenciales:
- Lenguaje claro y sencillo, dirigido a pacientes sin conocimientos médicos
- Formato de preguntas y respuestas donde sea apropiado
- Estructura estandarizada siguiendo las secciones oficiales
- Advertencias claramente resaltadas
- Instrucciones claras de dosificación y administración

ESTRUCTURA OFICIAL DEL PROSPECTO SEGÚN AEMPS:

1. NOMBRE DEL MEDICAMENTO
   [Nombre comercial, formulación y concentración]

2. QUÉ ES [MEDICAMENTO] Y PARA QUÉ SE UTILIZA
   - Descripción simple del grupo terapéutico
   - Indicaciones terapéuticas en lenguaje comprensible

3. ANTES DE TOMAR [MEDICAMENTO]
   - No tome [MEDICAMENTO] si... (contraindications)
   - Advertencias y precauciones
   - Uso de otros medicamentos
   - Toma de [MEDICAMENTO] con alimentos y bebidas
   - Embarazo y lactancia
   - Conducción y uso de máquinas
   - Información sobre excipientes

4. CÓMO TOMAR [MEDICAMENTO]
   - Posología detallada en lenguaje sencillo
   - Forma de administración
   - Duración del tratamiento
   - Si toma más [MEDICAMENTO] del que debe
   - Si olvidó tomar [MEDICAMENTO]
   - Si interrumpe el tratamiento con [MEDICAMENTO]

5. POSIBLES EFECTOS ADVERSOS
   - Clasificados por frecuencia
   - Explicados en términos comprensibles
   - Instrucciones sobre cuándo consultar al médico

6. CONSERVACIÓN DE [MEDICAMENTO]
   - Condiciones de conservación
   - Mantener fuera del alcance de los niños
   - Fecha de caducidad

7. INFORMACIÓN ADICIONAL
   - Composición
   - Aspecto y contenido del envase
   - Titular de la autorización y responsable de fabricación

Utiliza EXACTAMENTE esta estructura y estas secciones, utilizando el contenido del prospecto oficial proporcionado en los datos de CIMA. Si alguna sección no tiene información disponible, indícalo claramente.

Evita utilizar términos técnicos innecesarios. Cuando sea imprescindible usar un término médico complejo, explícalo brevemente de forma comprensible.
"""

    def __init__(self, openai_client: AsyncOpenAI):
        """Initialize the prospecto generator with OpenAI client"""
        self.openai_client = openai_client
        self.reference_cache = {}
        self.session = None
        self.max_tokens = 14000
        self.use_langgraph = True
        # Initialize tokenizer
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        # Active principle database - Spanish
        self.active_principles = [
           "ibuprofeno", "paracetamol", "omeprazol", "amoxicilina", "simvastatina",
            "enalapril", "metformina", "lorazepam", "diazepam", "fluoxetina",
            "atorvastatina", "tramadol", "naproxeno", "metamizol", "azitromicina",
            "aspirina", "acido acetilsalicilico", "salbutamol", "fluticasona",
            "amlodipino", "valsartan", "losartan", "dexametasona", "betametasona",
            "fentanilo", "morfina", "alendronato", "quetiapina", "risperidona",
            "levotiroxina", "ranitidina", "levofloxacino", "ciprofloxacino",
            "ondansetron", "prednisona", "hidrocortisona", "clonazepam",
            "melatonina", "warfarina", "acenocumarol", "alprazolam", "atenolol",
            "alopurinol", "amitriptilina", "diclofenaco", "loratadina", "cetirizina",
            "vitamina d", "calcio", "hierro", "insulina", "metronidazol",
            "minoxidil", "nolotil", "escitalopram", "bromazepam", "pantoprazol"
        ]

    def num_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a string"""
        return len(self.tokenizer.encode(text))

    async def get_session(self):
        """Get or create an aiohttp session with improved error handling"""
        if self.session is None or self.session.closed:
            # Using TCPConnector with proper settings for Streamlit environment
            connector = aiohttp.TCPConnector(
                ssl=True,  # Enable SSL verification
                limit=5,    # Lower connection limit to prevent resource exhaustion
                keepalive_timeout=30,  # Shorter keepalive period
                force_close=False      # Let the server control connection closing
            )
            timeout = aiohttp.ClientTimeout(
                total=60,    # Longer total timeout
                connect=20,  # Longer connect timeout
                sock_connect=20,
                sock_read=30
            )
            self.session = aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout,
                raise_for_status=False  # Don't raise exceptions for HTTP errors
            )
        return self.session

    def detect_prospecto_request(self, query: str) -> Dict[str, Any]:
        """
        Detect if a query is asking for a prospecto and extract medication info.
        
        Args:
            query: The query text
            
        Returns:
            Dictionary with detected information
        """
        query_lower = query.lower()
        
        # Enhanced prospecto request detection pattern
        prospecto_pattern = r'(?:redactar|generar|crear|elaborar|realizar?e?|escrib[ei]r|hac[ae]r|desarroll[ae]r|realiza(?:r|)|prepar(?:ar|a)|necesito|quiero)\s+(?:un|el|uns?|una?|)?(?:\s+|\s*)\b(?:prospecto|folleto|información para el paciente|leaflet|insert|pil)\b'
        is_prospecto = bool(re.search(prospecto_pattern, query_lower))
        
        # Check if this mentions "prospecto" and related terms
        prospecto_terms = ["prospecto", "folleto", "información para paciente", "indicaciones", "leaflet"]
        if not is_prospecto and any(term in query_lower for term in prospecto_terms):
            is_prospecto = True
            
        # Extract active principle
        active_principle = None
        for ap in self.active_principles:
            if ap in query_lower:
                active_principle = ap
                break
        
        # If active principle not found, try to extract from capitalization
        if not active_principle:
            # Look for compound active principles (e.g., "Hidrocortisona y Lidocaína")
            compound_pattern = r'([A-Z][a-z]+(?:\s[a-z]+)*)\s+[y]\s+([A-Z][a-z]+(?:\s[a-z]+)*)'
            compound_match = re.search(compound_pattern, query)
            if compound_match:
                active_principle = f"{compound_match.group(1)} {compound_match.group(2)}"
            else:
                # Look for capitalized words
                cap_words = re.findall(r'\b[A-Z][a-z]{2,}\b', query)
                if cap_words:
                    active_principle = cap_words[0]
                else:
                    # Just take the longest word as a guess
                    words = [w for w in query_lower.split() if len(w) > 4 and not any(x in w for x in ['como', 'para', 'sobre', 'cual', 'este', 'esta', 'prospecto', 'generar', 'crear'])]
                    if words:
                        active_principle = max(words, key=len)
        
        # Check for medication names like "MINOXIDIL BIORGA"
        uppercase_names = re.findall(r'\b[A-Z]{2,}\s+[A-Z]{2,}\b', query.upper())
        
        # Extract concentration if present
        concentration_pattern = r'(\d+(?:[,.]\d+)?\s*(?:%|mg|g|ml|mcg|UI|unidades)|\d+\s*(?:mg)?[/](?:ml|g))'
        concentration_match = re.search(concentration_pattern, query)
        concentration = concentration_match.group(0) if concentration_match else None
        
        return {
            "is_prospecto": is_prospecto,
            "active_principle": active_principle,
            "uppercase_names": uppercase_names,
            "concentration": concentration
        }

    def clean_html(self, html_content: str) -> str:
        """
        Clean HTML content to get plain text, preserving basic structure.
        
        Args:
            html_content: HTML content to clean
            
        Returns:
            Cleaned text
        """
        if not html_content or html_content == "No disponible":
            return "No disponible"
            
        try:
            # Use BeautifulSoup to clean the HTML if available
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, "html.parser")
                # Convert <br> and <p> to newlines
                for br in soup.find_all("br"):
                    br.replace_with("\n")
                for p in soup.find_all("p"):
                    p.append("\n\n")
                # Convert headings to capitalized text with newlines
                for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    heading_text = heading.get_text().strip().upper()
                    heading.replace_with(f"\n\n{heading_text}\n\n")
                # Convert lists to bullet points
                for ul in soup.find_all("ul"):
                    for li in ul.find_all("li"):
                        li_text = li.get_text().strip()
                        li.replace_with(f"• {li_text}\n")
                # Get the cleaned text
                return soup.get_text().replace("\n\n\n\n", "\n\n").replace("\n\n\n", "\n\n").strip()
            except ImportError:
                # Fallback to regex if BeautifulSoup is not available
                text = re.sub(r'<br\s*/?>', '\n', html_content)
                text = re.sub(r'<p\b[^>]*>(.*?)</p>', r'\1\n\n', text)
                text = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r'\n\n\1\n\n', text)
                text = re.sub(r'<li\b[^>]*>(.*?)</li>', r'• \1\n', text)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text)
                return text.strip()
        except Exception as e:
            logger.error(f"Error cleaning HTML: {str(e)}")
            # Simple fallback
            return re.sub(r'<[^>]+>', ' ', html_content).strip()

    async def get_medication_context(self, query: str) -> str:
        """
        Gets medication context for prospecto generation using CIMA data.
        Prioritizes the official prospecto content from the API.
        
        Args:
            query: Query about medication prospecto
            
        Returns:
            Context text with medication information
        """
        # Use a search implementation to find medication data
        search_implementation = MedicationSearchGraph()
        results, quality, query_intent = await search_implementation.execute_search(query)
        
        if not results:
            return "No se encontraron medicamentos relevantes para esta consulta."
        
        # Get the most relevant medication with proper error handling
        try:
            # Check if the first result is a dictionary
            top_med = results[0]
            if not isinstance(top_med, dict):
                logger.error(f"Unexpected result type: {type(top_med)}")
                return "Error: Resultado en formato inesperado."
            
            # Now safely extract the nregistro
            nregistro = top_med.get("nregistro")
            if not nregistro:
                logger.error("No nregistro found in top result")
                return "No se pudo encontrar el número de registro del medicamento."
            
            # Get medication details
            session = await self.get_session()
            url = f"{self.base_url}/medicamento"
            
            async with session.get(url, params={"nregistro": nregistro}) as response:
                if response.status != 200:
                    return f"Error al obtener información del medicamento: {response.status}"
                    
                basic_info = await response.json()
            
            # ENHANCED: Try multiple sources to get the best prospecto content
            prospecto_content = "No disponible"
            prospecto_sources = [
                # Official API endpoint for segmented prospecto
                (f"{self.base_url}/docSegmentado/contenido/2", {"nregistro": nregistro}),
                # Alternative direct HTML sources
                (f"https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html", None),
                (f"https://cima.aemps.es/cima/dochtml/p/{nregistro}/Prospecto_{nregistro}.html", None)
            ]
            
            # Try each source until we get valid content
            for source_url, params in prospecto_sources:
                try:
                    if params:
                        async with session.get(source_url, params=params) as response:
                            if response.status == 200:
                                try:
                                    data = await response.json()
                                    if isinstance(data, dict) and "contenido" in data:
                                        prospecto_content = data["contenido"]
                                        # If we got content, clean and break
                                        if prospecto_content and prospecto_content != "No disponible":
                                            logger.info(f"Got prospecto content from {source_url} with params")
                                            prospecto_content = self.clean_html(prospecto_content)
                                            break
                                except:
                                    # If not JSON, try as HTML
                                    raw_content = await response.text()
                                    if len(raw_content) > 100:  # Basic validation
                                        prospecto_content = self.clean_html(raw_content)
                                        logger.info(f"Got prospecto content from {source_url} with params as HTML")
                                        break
                    else:
                        # Direct URL without params
                        async with session.get(source_url) as response:
                            if response.status == 200:
                                raw_content = await response.text()
                                if len(raw_content) > 100:  # Basic validation
                                    prospecto_content = self.clean_html(raw_content)
                                    logger.info(f"Got prospecto content from {source_url} direct")
                                    break
                except Exception as e:
                    logger.warning(f"Error accessing {source_url}: {str(e)}")
                    continue
            
            # Get structured information for better prospecto generation
            structured_sections = {}
            try:
                # Try to get structured sections if available
                sections_to_fetch = {
                    "indicaciones": "4.1",
                    "posologia": "4.2",
                    "contraindicaciones": "4.3",
                    "advertencias": "4.4",
                    "interacciones": "4.5",
                    "embarazo": "4.6",
                    "efectos_adversos": "4.8",
                    "excipientes": "6.1"
                }
                
                for section_key, section_id in sections_to_fetch.items():
                    api_section_id = section_id.replace(".", "")
                    tech_url = f"{self.base_url}/docSegmentado/contenido/1"
                    params = {"nregistro": nregistro, "seccion": api_section_id}
                    
                    try:
                        async with session.get(tech_url, params=params) as response:
                            if response.status == 200:
                                section_data = await response.json()
                                if "contenido" in section_data:
                                    content = section_data["contenido"]
                                    if content and content != "No disponible":
                                        structured_sections[section_key] = self.clean_html(content)
                    except Exception as e:
                        logger.warning(f"Error fetching section {section_key}: {str(e)}")
            except Exception as e:
                logger.warning(f"Error fetching structured sections: {str(e)}")
            
            # Format the context with emphasis on prospecto structure
            context = f"""
INFORMACIÓN DEL MEDICAMENTO:
- Nombre: {basic_info.get('nombre', 'No disponible')}
- Número de registro: {nregistro}
- Principio(s) activo(s): {basic_info.get('pactivos', 'No disponible')}
- Laboratorio titular: {basic_info.get('labtitular', 'No disponible')}

=== PROSPECTO OFICIAL DEL MEDICAMENTO ===
{prospecto_content}

"""
            
            # Add structured sections if available and prospecto is not detailed enough
            if structured_sections and (prospecto_content == "No disponible" or len(prospecto_content) < 500):
                context += "\n=== INFORMACIÓN ADICIONAL ESTRUCTURADA (PARA COMPLETAR EL PROSPECTO) ===\n"
                for section_key, content in structured_sections.items():
                    section_name = section_key.upper().replace("_", " ")
                    context += f"\n{section_name}:\n{content}\n"
            
            # Add document links
            context += f"""
URL FICHA TÉCNICA:
https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html

URL PROSPECTO:
https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html
"""
            return context
            
        except Exception as e:
            logger.error(f"Error getting medication context: {str(e)}")
            return f"Error al obtener información del medicamento: {str(e)}"

    async def generate_prospecto(self, query: str) -> Dict[str, str]:
        """
        Generate a complete medication prospecto based on CIMA data.
        The output follows official AEMPS patient information leaflet standards.
        
        Args:
            query: Query about medication prospecto
            
        Returns:
            Dictionary with prospecto response and context
        """
        # Check if this is a prospecto request
        query_info = self.detect_prospecto_request(query)
        
        if not query_info["is_prospecto"]:
            return {
                "prospecto": "La consulta no parece estar solicitando un prospecto. Por favor, reformule su consulta indicando claramente que desea generar un prospecto para un medicamento específico.",
                "context": "",
                "medication": ""
            }
        
        # Get context for the medication
        context = await self.get_medication_context(query)
        
        if "Error" in context or "No se encontraron" in context:
            return {
                "prospecto": f"No se pudo generar el prospecto: {context}",
                "context": context,
                "medication": query_info.get("active_principle", "")
            }
        
        # Create a prompt for generating the prospecto
        prompt = f"""
Genera un prospecto de medicamento en formato AEMPS oficial para pacientes a partir de la siguiente información:

CONSULTA DEL USUARIO:
{query}

CONTEXTO DEL MEDICAMENTO:
{context}

INSTRUCCIONES ESPECÍFICAS:
1. Utiliza EXCLUSIVAMENTE la estructura de un prospecto de AEMPS oficial (para pacientes)
2. Mantén un lenguaje sencillo y comprensible para cualquier paciente
3. Usa formato de preguntas y respuestas donde sea apropiado (ej: "¿Qué es X y para qué se utiliza?")
4. Respeta el formato de secciones numeradas según las directrices oficiales
5. Utiliza viñetas (•) para listar elementos cuando sea apropiado
6. IMPORTANTE: No inventes información; si no hay datos para alguna sección, indícalo claramente
7. Prioriza siempre la información del prospecto oficial que está en el contexto

RECUERDA: Un prospecto es un documento informativo para PACIENTES, no para profesionales sanitarios.
"""

        try:
            # Generate the prospecto using OpenAI
            response = await self.openai_client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4  # Lower temperature for more adherence to the format
            )
            
            prospecto = response.choices[0].message.content
            
            # Extract medication name from context
            medication_name = "No disponible"
            match = re.search(r'Nombre: ([^\n]+)', context)
            if match:
                medication_name = match.group(1)
            
            return {
                "prospecto": prospecto,
                "context": context,
                "medication": medication_name
            }
        except Exception as e:
            logger.error(f"Error generating prospecto: {str(e)}")
            return {
                "prospecto": f"Error al generar el prospecto: {str(e)}",
                "context": context,
                "medication": query_info.get("active_principle", "")
            }
    
    async def close(self):
        """Close the aiohttp session to free resources with proper cleanup"""
        if self.session and not self.session.closed:
            try:
                await self.session.close()
                # Give the event loop time to clean up connections
                await asyncio.sleep(0.25)
            except Exception as e:
                logger.error(f"Error closing session: {str(e)}")
                # Ensure session is marked as closed even if there was an error
                self.session = None