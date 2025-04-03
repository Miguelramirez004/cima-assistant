"""
Prospecto generator module for creating medication package inserts (patient information leaflets)
following the official AEMPS format for Spanish medications.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProspectoGenerator:
    """
    Generator for Spanish medication package inserts (prospectos) using CIMA API data.
    Creates patient-friendly medication information in official AEMPS format.
    """
    openai_client: AsyncOpenAI
    reference_cache: Dict[str, List[Dict]] = field(default_factory=dict)
    session: aiohttp.ClientSession = None
    max_tokens: int = 14000  # Leave room for prompt and response
    use_langgraph: bool = True  # Use the LangGraph search by default
    base_url: str = Config.CIMA_BASE_URL
    
    # System prompt specifically optimized for authentic Spanish prospecto generation
    system_prompt = """Eres un experto en redacción de prospectos de medicamentos siguiendo estrictamente el formato oficial de la AEMPS española (Agencia Española de Medicamentos y Productos Sanitarios).

IMPORTANTE: Un prospecto NO es lo mismo que una formulación magistral ni una ficha técnica. Un prospecto es un documento oficial dirigido a PACIENTES que acompaña a los medicamentos y explica, en lenguaje sencillo y accesible, toda la información necesaria para el uso correcto del medicamento.

Debes escribir en tono cercano y comprensible para un paciente promedio, evitando tecnicismos innecesarios y explicando los términos médicos complejos cuando sea necesario.

ESTRUCTURA OFICIAL DEL PROSPECTO SEGÚN AEMPS:

1. PROSPECTO: INFORMACIÓN PARA EL USUARIO/PACIENTE
   [Nombre comercial, sustancia activa, formulación y dosificación]

2. QUÉ ES [MEDICAMENTO] Y PARA QUÉ SE UTILIZA
   - Descripción sencilla del grupo terapéutico
   - Enfermedades o condiciones para las que está indicado
   - Breve explicación de cómo funciona

3. ANTES DE TOMAR [MEDICAMENTO]
   - No tome [MEDICAMENTO] si: (contraindications)
   - Advertencias y precauciones 
   - Uso de [MEDICAMENTO] con otros medicamentos
   - Toma de [MEDICAMENTO] con alimentos y bebidas
   - Embarazo, lactancia y fertilidad
   - Conducción y uso de máquinas
   - [MEDICAMENTO] contiene... (información sobre excipientes)

4. CÓMO TOMAR/USAR [MEDICAMENTO]
   - Instrucciones precisas de dosificación por grupo de edad
   - Método de administración (cómo tomar correctamente)
   - Duración del tratamiento
   - Si toma más [MEDICAMENTO] del que debe
   - Si olvidó tomar [MEDICAMENTO]
   - Si interrumpe el tratamiento con [MEDICAMENTO]

5. POSIBLES EFECTOS ADVERSOS
   - Listado por frecuencia (muy frecuentes, frecuentes, poco frecuentes, raros, muy raros)
   - Instrucciones sobre qué hacer si aparecen efectos adversos
   - Frase estándar para reportar efectos adversos

6. CONSERVACIÓN DE [MEDICAMENTO]
   - Condiciones de almacenamiento
   - Mantener fuera de la vista y alcance de los niños
   - Fecha de caducidad
   - Instrucciones de eliminación

7. CONTENIDO DEL ENVASE E INFORMACIÓN ADICIONAL
   - Composición (principio activo y excipientes)
   - Aspecto y contenido del envase
   - Titular de la autorización y responsable de fabricación
   - Fecha de última revisión del prospecto

IMPORTANTE:
1. Utiliza un lenguaje directo, con frases cortas y palabras comunes
2. Usa la segunda persona ("usted") para dirigirte al paciente
3. Incluye los encabezados exactamente como se muestran arriba, en formato de preguntas cuando corresponda
4. Destaca las advertencias importantes en formato negrita
5. Usa viñetas (•) para listas de recomendaciones, efectos adversos, etc.
6. Mantén la objetividad y precisión de la información médica, simplificándola pero sin perder exactitud
7. Sigue estrictamente los datos de CIMA y el prospecto original proporcionado

Asegúrate de usar el prospecto oficial de CIMA como referencia principal, ya que éste ya contiene la estructura y lenguaje apropiados para pacientes.
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
            # Don't use global timeout - use per-request timeout to avoid context issues
            self.session = aiohttp.ClientSession(
                connector=connector,
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
        
        # Enhanced prospecto request detection patterns
        prospecto_pattern = r'(?:redactar|generar|crear|elaborar|realizar?e?|escrib[ei]r|hac[ae]r|desarroll[ae]r|realiza(?:r|)|prepar(?:ar|a))\s+(?:un|el|uns?|una?)?\s+prospecto'
        leaflet_pattern = r'(?:folleto|hoja informativa|informaci[óo]n para (?:el|la) paciente|instrucciones)'
        
        # Check if this is a prospecto request
        is_prospecto = bool(re.search(prospecto_pattern, query_lower)) or bool(re.search(leaflet_pattern, query_lower))
        
        # Also check if this mentions "prospecto" directly
        if not is_prospecto and "prospecto" in query_lower:
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
                    words = [w for w in query_lower.split() if len(w) > 4 and not any(x in w for x in ['como', 'para', 'sobre', 'cual', 'este', 'esta', 'prospecto', 'generar'])]
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

    async def get_medication_context(self, query: str) -> str:
        """
        Gets medication context for prospecto generation with emphasis on patient-oriented content.
        
        Args:
            query: Query about medication prospecto
            
        Returns:
            Context text with medication information, prioritizing prospecto content
        """
        # Use a search implementation to find medication data
        search_implementation = MedicationSearchGraph()
        results, quality, query_intent = await search_implementation.execute_search(query)
        
        if not results:
            return "No se encontraron medicamentos relevantes para esta consulta."
        
        # Get the most relevant medication - with proper error handling
        try:
            # Check if the first result is a dictionary (as it should be)
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
            
            # Get basic info with explicit timeout
            basic_info = {}
            try:
                url = f"{self.base_url}/medicamento"
                async with session.get(url, params={"nregistro": nregistro}, timeout=30) as response:
                    if response.status != 200:
                        logger.warning(f"Error getting basic info: {response.status}")
                    else:
                        basic_info = await response.json()
            except Exception as e:
                logger.error(f"Error fetching basic info: {str(e)}")
                basic_info = {"nombre": f"Medicamento {nregistro}", "pactivos": "No disponible", "labtitular": "No disponible"}
            
            # PRIORITIZE prospecto content specifically
            prospecto_content = await self._get_prospecto_content(nregistro)
                
            # Create the context with clear separation between basic info and prospecto content
            context = f"""
INFORMACIÓN BÁSICA DEL MEDICAMENTO:
- Nombre: {basic_info.get('nombre', 'No disponible')}
- Número de registro: {nregistro}
- Principio(s) activo(s): {basic_info.get('pactivos', 'No disponible')}
- Laboratorio titular: {basic_info.get('labtitular', 'No disponible')}
- Estado de comercialización: {"Comercializado" if basic_info.get('comerc') else "No comercializado"}

----PROSPECTO OFICIAL----
{prospecto_content}
----FIN PROSPECTO OFICIAL----

Enlaces de referencia:
- URL Ficha Técnica: https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html
- URL Prospecto: https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html
"""
            return context
            
        except Exception as e:
            logger.error(f"Error getting medication context: {str(e)}")
            return f"Error al obtener información del medicamento: {str(e)}"

    async def _get_prospecto_content(self, nregistro: str) -> str:
        """
        Get prospecto content with multiple fallback methods to ensure we get the patient information.
        
        Args:
            nregistro: Registration number of the medication
            
        Returns:
            Prospecto content as text
        """
        session = await self.get_session()
        prospecto_content = "No disponible"
        
        # Method 1: Try to get content using the docSegmentado API (most reliable for structured content)
        try:
            prospecto_url = f"{self.base_url}/docSegmentado/contenido/2"
            # Use explicit timeout for each request to avoid context errors
            async with session.get(prospecto_url, params={"nregistro": nregistro}, timeout=30) as response:
                if response.status == 200:
                    prospecto_data = await response.json()
                    if isinstance(prospecto_data, dict) and "contenido" in prospecto_data:
                        content = prospecto_data.get("contenido", "")
                        if content and content != "No disponible":
                            # Clean HTML for better readability if needed
                            prospecto_content = self._clean_html(content)
                            return prospecto_content
        except Exception as e:
            logger.warning(f"Error getting prospecto via primary method: {str(e)}")
        
        # Method 2: Try to get HTML directly from web version
        try:
            urls_to_try = [
                f"https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html",
                f"https://cima.aemps.es/cima/dochtml/p/{nregistro}/Prospecto_{nregistro}.html"
            ]
            
            for url in urls_to_try:
                # Add explicit timeout to each request to avoid context errors
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        if html_content and len(html_content) > 100:
                            # Extract the main content from HTML
                            main_content = self._extract_content_from_html(html_content)
                            if main_content:
                                return main_content
        except Exception as e:
            logger.warning(f"Error getting prospecto via HTML: {str(e)}")
                
        # Fallback: If no prospecto was found, attempt to create placeholder with any available info
        if prospecto_content == "No disponible":
            # Try getting sections from ficha técnica that could be reformatted
            try:
                useful_sections = ["indicaciones", "posologia_procedimiento", "contraindicaciones", "advertencias"]
                section_content = []
                
                for section in useful_sections:
                    section_url = f"{self.base_url}/docSegmentado/contenido/1"
                    section_id = {"indicaciones": "41", "posologia_procedimiento": "42", 
                                 "contraindicaciones": "43", "advertencias": "44"}.get(section, section)
                    
                    # Add explicit timeout to each request to avoid context errors
                    async with session.get(section_url, params={"nregistro": nregistro, "seccion": section_id}, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            if isinstance(data, dict) and "contenido" in data:
                                content = data.get("contenido", "")
                                if content and content != "No disponible":
                                    section_content.append(f"INFORMACIÓN DE {section.upper()}:\n{self._clean_html(content)}")
                
                if section_content:
                    return "INFORMACIÓN RECOPILADA DE LA FICHA TÉCNICA (No se encontró el prospecto completo):\n\n" + "\n\n".join(section_content)
            except Exception as e:
                logger.warning(f"Error getting fallback sections: {str(e)}")
        
        return prospecto_content
        
    def _clean_html(self, html_content: str) -> str:
        """
        Clean HTML tags from content for better readability.
        
        Args:
            html_content: HTML content to clean
            
        Returns:
            Cleaned text content
        """
        # Remove HTML tags
        cleaned = re.sub(r'<[^>]+>', ' ', html_content)
        # Fix spacing issues
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # Add back some formatting for headings
        cleaned = re.sub(r'([0-9]\.)([A-Z])', r'\n\1 \2', cleaned)
        # Add back paragraph breaks
        cleaned = re.sub(r'(\. )([A-Z])', r'.\n\2', cleaned)
        
        return cleaned
        
    def _extract_content_from_html(self, html_content: str) -> str:
        """
        Extract main content from HTML prospecto.
        
        Args:
            html_content: Full HTML content
            
        Returns:
            Extracted main content
        """
        # Try to find the main content section
        main_content_match = re.search(r'<div[^>]*?(?:id=["\'](content|main|prospecto)["\']|class=["\'](content|main|prospecto)["\'])[^>]*>(.*?)</div>(?:</div>|<footer)', html_content, re.DOTALL)
        if main_content_match:
            # Extract and clean content
            content = main_content_match.group(3)
            return self._clean_html(content)
            
        # If we can't find a specific content div, extract text between title and footer
        body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.DOTALL)
        if body_match:
            body_content = body_match.group(1)
            # Remove header, navigation, footer, etc.
            cleaned = re.sub(r'<header.*?</header>', '', body_content, flags=re.DOTALL)
            cleaned = re.sub(r'<nav.*?</nav>', '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r'<footer.*?</footer>', '', cleaned, flags=re.DOTALL)
            # Clean remaining HTML tags
            return self._clean_html(cleaned)
            
        # Last resort: just clean all HTML tags
        return self._clean_html(html_content)

    async def generate_prospecto(self, query: str) -> Dict[str, str]:
        """
        Generate a complete medication prospecto in official AEMPS format.
        
        Args:
            query: Query about medication prospecto
            
        Returns:
            Dictionary with prospecto response and context
        """
        # Check if this is a prospecto request
        query_info = self.detect_prospecto_request(query)
        
        if not query_info["is_prospecto"] and not query_info["active_principle"]:
            return {
                "prospecto": "La consulta no parece estar solicitando un prospecto. Por favor, reformule su consulta indicando claramente que desea generar un prospecto para un medicamento específico.",
                "context": "",
                "medication": ""
            }
        
        # Get context for the medication - with enhanced prospecto retrieval
        context = await self.get_medication_context(query)
        
        if "Error" in context or "No se encontraron" in context:
            return {
                "prospecto": f"No se pudo generar el prospecto: {context}",
                "context": context,
                "medication": query_info.get("active_principle", "")
            }
        
        # Create an enhanced prompt that emphasizes following the AEMPS prospecto format
        prompt = f"""
Genera un prospecto oficial siguiendo exactamente el formato de la AEMPS para la siguiente consulta:

CONSULTA DEL USUARIO:
{query}

DATOS DEL MEDICAMENTO:
{context}

INSTRUCCIONES ESPECÍFICAS:
1. Redacta el prospecto siguiendo exactamente la estructura oficial de la AEMPS para prospectos de medicamentos
2. Utiliza como referencia principal el contenido del prospecto oficial de CIMA que se incluye en el contexto
3. Utiliza un lenguaje simple, claro y dirigido al paciente, NO al profesional sanitario
4. Incluye todos los apartados que aparecen en los prospectos oficiales, con sus encabezados exactos
5. Destaca las advertencias importantes en negrita como se hace en los prospectos reales
6. Usa viñetas para las listas de efectos adversos, precauciones, etc.
7. El resultado debe ser un prospecto auténtico idéntico en estructura y formato a los que acompañan a medicamentos en España

NO incluyas información técnica innecesaria que confundiría a un paciente. El prospecto debe ser accesible para cualquier persona sin conocimientos médicos.
"""

        try:
            # Generate the prospecto using OpenAI
            response = await self.openai_client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5  # Slightly lower temperature for more consistency
            )
            
            prospecto = response.choices[0].message.content
            
            # Extract medication name from context
            medication_name = "No disponible"
            match = re.search(r'Nombre: ([^\n]+)', context)
            if match:
                medication_name = match.group(1)
            
            # Post-process the prospecto to ensure proper formatting
            prospecto = self._format_prospecto(prospecto, medication_name)
            
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
    
    def _format_prospecto(self, raw_prospecto: str, medication_name: str) -> str:
        """
        Apply final formatting to ensure prospecto follows AEMPS conventions.
        
        Args:
            raw_prospecto: Generated prospecto text
            medication_name: Medication name for title
            
        Returns:
            Properly formatted prospecto
        """
        # Ensure the prospecto starts with the standard header if it doesn't already
        if not raw_prospecto.strip().startswith("PROSPECTO:"):
            raw_prospecto = f"PROSPECTO: INFORMACIÓN PARA EL PACIENTE\n\n{medication_name}\n\n" + raw_prospecto
            
        # Ensure section headings are properly formatted
        standard_sections = [
            "QUÉ ES", "ANTES DE", "CÓMO TOMAR", "POSIBLES EFECTOS", 
            "CONSERVACIÓN", "CONTENIDO DEL ENVASE"
        ]
        
        for section in standard_sections:
            # Find headings without proper formatting and add it
            pattern = re.compile(f"([^#\n])(\\d+\\. {section})", re.IGNORECASE)
            raw_prospecto = pattern.sub(r"\1\n\n\2", raw_prospecto)
            
            # Ensure headings are in bold
            pattern = re.compile(f"(\\d+\\. {section}[^\n]*)", re.IGNORECASE)
            raw_prospecto = pattern.sub(r"**\1**", raw_prospecto)
        
        # Add blank lines before subheadings for readability
        raw_prospecto = re.sub(r'(\n)(-|\•) ', r'\n\n$2 ', raw_prospecto)
        
        # Ensure warning sections are in bold
        warnings = ["No tome", "Advertencias y precauciones", "Si toma más", "Si olvidó tomar"]
        for warning in warnings:
            pattern = re.compile(f"({warning}[^:\n]*:)", re.IGNORECASE)
            raw_prospecto = pattern.sub(r"**\1**", raw_prospecto)
        
        # Make sure effect frequency sections are emphasized
        frequencies = ["Muy frecuentes", "Frecuentes", "Poco frecuentes", "Raros", "Muy raros"]
        for freq in frequencies:
            pattern = re.compile(f"({freq}[^:\n]*:)", re.IGNORECASE)
            raw_prospecto = pattern.sub(r"*\1*", raw_prospecto)
            
        return raw_prospecto
    
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