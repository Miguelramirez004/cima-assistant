"""
Prospecto generator module for creating medication package inserts (prospectos)
following official AEMPS guidelines for patient information leaflets.
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
import html
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProspectoGenerator:
    """
    Generator for medication package inserts (prospectos) using CIMA data,
    following the official AEMPS guidelines for patient information leaflets.
    """
    openai_client: AsyncOpenAI
    reference_cache: Dict[str, List[Dict]] = field(default_factory=dict)
    session: aiohttp.ClientSession = None
    max_tokens: int = 14000  # Leave room for prompt and response
    use_langgraph: bool = True  # Use the LangGraph search by default
    base_url: str = Config.CIMA_BASE_URL
    
    # System prompt specifically for prospecto generation matching AEMPS standards
    system_prompt = """Eres un experto en redacción de prospectos de medicamentos siguiendo estrictamente el formato oficial de la AEMPS española (Agencia Española de Medicamentos y Productos Sanitarios).

IMPORTANTE: Un prospecto NO es lo mismo que una formulación magistral ni una ficha técnica. Un prospecto es un documento dirigido a PACIENTES que acompaña a los medicamentos y explica, en lenguaje sencillo y accesible, toda la información necesaria para el uso correcto del medicamento.

Tu tarea es generar un prospecto que siga exactamente la estructura y formato oficial de la AEMPS, utilizando un tono cercano y comprensible para pacientes, evitando tecnicismos innecesarios y explicando los términos médicos cuando sea imprescindible.

ESTRUCTURA OFICIAL DEL PROSPECTO SEGÚN AEMPS:

1. NOMBRE DEL MEDICAMENTO
   [Nombre completo con forma farmacéutica y concentración]

2. QUÉ ES [MEDICAMENTO] Y PARA QUÉ SE UTILIZA
   [Descripción sencilla del grupo terapéutico y para qué condiciones se usa]

3. ANTES DE TOMAR [MEDICAMENTO]
   - No tome [MEDICAMENTO]... [Contraindications]
   - Advertencias y precauciones
   - Uso de otros medicamentos
   - Toma de [MEDICAMENTO] con alimentos y bebidas
   - Embarazo y lactancia
   - Conducción y uso de máquinas
   - Información importante sobre algunos de los componentes de [MEDICAMENTO]

4. CÓMO TOMAR [MEDICAMENTO]
   - Instrucciones claras sobre dosis
   - Forma de administración
   - Duración del tratamiento
   - Si toma más [MEDICAMENTO] del que debiera
   - Si olvidó tomar [MEDICAMENTO]
   - Si interrumpe el tratamiento con [MEDICAMENTO]

5. POSIBLES EFECTOS ADVERSOS
   - Clasificados por frecuencia (muy frecuentes, frecuentes, poco frecuentes, raros, muy raros)
   - Instrucciones sobre qué hacer si aparecen efectos adversos

6. CONSERVACIÓN DE [MEDICAMENTO]
   - Condiciones de conservación
   - Caducidad y qué hacer al respecto
   - Información sobre eliminación

7. INFORMACIÓN ADICIONAL
   - Composición (principio activo y excipientes)
   - Aspecto del producto y contenido del envase
   - Titular de la autorización de comercialización y responsable de la fabricación

Debes basar el prospecto en la información proporcionada en el contexto, priorizando el contenido del prospecto original siempre que esté disponible. Mantén un lenguaje accesible y amigable para el paciente, utilizando frases cortas y directas.

Utiliza con frecuencia estructuras como:
- Listas con viñetas para mayor claridad
- Frases cortas y directas
- Preguntas directas como encabezados
- Instrucciones específicas utilizando verbos imperativos

Evita:
- Términos médicos complejos sin explicación
- Oraciones subordinadas extensas 
- Información excesivamente técnica
- Referencias a estudios clínicos complejos

Sigue fielmente la estructura AEMPS indicada, manteniendo los apartados numerados y en el orden correcto.
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
        
        # Expanded prospecto request detection pattern
        prospecto_pattern = r'(?:redactar|generar|crear|elaborar|realizar?e?|escrib[ei]r|hac[ae]r|desarroll[ae]r|realiza(?:r|)|prepar(?:ar|a))\s+(?:un|el|uns?|una?)?\s+prospecto'
        is_prospecto = bool(re.search(prospecto_pattern, query_lower))
        
        # Check if this explicitly mentions "prospecto" 
        if not is_prospecto and "prospecto" in query_lower:
            is_prospecto = True
            
        # Extract active principle with improved pattern matching
        active_principle = None
        # Try direct matching first
        for ap in self.active_principles:
            # Use word boundary to avoid partial matches
            if re.search(r'\b' + re.escape(ap) + r'\b', query_lower):
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
                    words = [w for w in query_lower.split() if len(w) > 4 and not any(x in w for x in ['como', 'para', 'sobre', 'cual', 'este', 'esta', 'prospecto'])]
                    if words:
                        active_principle = max(words, key=len)
        
        # Check for medication names like "MINOXIDIL BIORGA"
        uppercase_names = re.findall(r'\b[A-Z]{2,}\s+[A-Z]{2,}\b', query.upper())
        
        # Extract concentration with improved pattern matching
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
        Gets medication context for prospecto generation using CIMA data,
        prioritizing the actual prospecto content.
        
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
        
        # Get the most relevant medication - with proper type checking
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
            
            # Get basic medication details
            session = await self.get_session()
            url = f"{self.base_url}/medicamento"
            
            async with session.get(url, params={"nregistro": nregistro}) as response:
                if response.status != 200:
                    return f"Error al obtener información del medicamento: {response.status}"
                    
                basic_info = await response.json()
            
            # ENHANCED: Get the full prospecto content directly - this is critical
            # Use multiple methods to ensure we get the prospecto data
            prospecto_content = await self._get_full_prospecto(nregistro)
            
            # Format the context with a clear PROSPECTO section
            context = f"""
INFORMACIÓN BÁSICA DEL MEDICAMENTO:
- Nombre: {basic_info.get('nombre', 'No disponible')}
- Número de registro: {nregistro}
- Principio(s) activo(s): {basic_info.get('pactivos', 'No disponible')}
- Laboratorio titular: {basic_info.get('labtitular', 'No disponible')}

CONTENIDO DEL PROSPECTO OFICIAL:
{prospecto_content}

URL FICHA TÉCNICA:
https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html

URL PROSPECTO:
https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html
"""
            return context
            
        except Exception as e:
            logger.error(f"Error getting medication context: {str(e)}")
            return f"Error al obtener información del medicamento: {str(e)}"

    async def _get_full_prospecto(self, nregistro: str) -> str:
        """
        Enhanced method to get the full prospecto content using multiple approaches
        to ensure we get the best possible data.
        
        Args:
            nregistro: Registration number
            
        Returns:
            Formatted prospecto content
        """
        session = await self.get_session()
        prospecto_content = "No disponible"
        
        # Try multiple methods to get the prospecto
        methods = [
            self._get_prospecto_from_api,
            self._get_prospecto_from_html,
            self._get_prospecto_from_pdf_url
        ]
        
        for method in methods:
            try:
                result = await method(nregistro, session)
                if result and result != "No disponible" and len(result) > 100:
                    # Clean up the content - remove excessive whitespace and HTML
                    cleaned_content = self._clean_prospecto_content(result)
                    if cleaned_content and len(cleaned_content) > 100:
                        logger.info(f"Successfully retrieved prospecto with method {method.__name__}")
                        return cleaned_content
            except Exception as e:
                logger.warning(f"Error retrieving prospecto with method {method.__name__}: {str(e)}")
        
        # If all methods failed, return fallback message
        return "El prospecto oficial no está disponible. Se generará un prospecto basado en la información general del medicamento."

    async def _get_prospecto_from_api(self, nregistro: str, session: aiohttp.ClientSession) -> str:
        """Get prospecto from CIMA API endpoint"""
        prospecto_url = f"{self.base_url}/docSegmentado/contenido/2"
        
        async with session.get(prospecto_url, params={"nregistro": nregistro}) as response:
            if response.status != 200:
                return "No disponible"
                
            try:
                prospecto_data = await response.json()
                content = prospecto_data.get("contenido", "No disponible")
                if content and content != "No disponible":
                    return content
                return "No disponible"
            except:
                return "No disponible"

    async def _get_prospecto_from_html(self, nregistro: str, session: aiohttp.ClientSession) -> str:
        """Get prospecto from HTML document URL"""
        prospecto_urls = [
            f"https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html",
            f"https://cima.aemps.es/cima/dochtml/p/{nregistro}/Prospecto_{nregistro}.html"
        ]
        
        for url in prospecto_urls:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        
                        # Use BeautifulSoup to extract the text
                        try:
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(html_content, 'html.parser')
                            
                            # Remove scripts, styles and other unwanted elements
                            for element in soup(["script", "style", "nav", "footer", "header"]):
                                element.decompose()
                                
                            # Get the main content
                            main_content = soup.find("div", {"class": ["page", "content", "main-content"]})
                            if main_content:
                                text = main_content.get_text(separator="\n")
                            else:
                                text = soup.get_text(separator="\n")
                                
                            # Clean up the text
                            lines = [line.strip() for line in text.splitlines() if line.strip()]
                            text = "\n".join(lines)
                            
                            return text
                        except:
                            # If BeautifulSoup fails, just return basic HTML
                            return html_content
            except:
                continue
                
        return "No disponible"

    async def _get_prospecto_from_pdf_url(self, nregistro: str, session: aiohttp.ClientSession) -> str:
        """
        Get prospecto from PDF URL
        Note: This doesn't download/parse the PDF, it just returns the URL
        for reference purposes
        """
        pdf_url = f"https://cima.aemps.es/cima/pdfs/p/{nregistro}/P_{nregistro}.pdf"
        
        try:
            async with session.head(pdf_url) as response:
                if response.status == 200:
                    return f"Prospecto disponible en PDF: {pdf_url}"
        except:
            pass
            
        return "No disponible"

    def _clean_prospecto_content(self, content: str) -> str:
        """
        Clean up prospecto content by removing HTML tags and excess whitespace
        
        Args:
            content: Raw prospecto content
            
        Returns:
            Cleaned prospecto content
        """
        # First try to clean with BeautifulSoup if it's HTML content
        if "<html" in content.lower() or "<body" in content.lower() or "<div" in content.lower():
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(["script", "style"]):
                    element.decompose()
                
                # Get text with line breaks
                text = soup.get_text(separator="\n")
                
                # Clean up whitespace
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                text = "\n".join(lines)
                
                return text
            except Exception as e:
                logger.warning(f"Error cleaning HTML content: {str(e)}")
        
        # Fallback to basic cleaning if BeautifulSoup fails or content isn't HTML
        # Strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', content)
        
        # Decode HTML entities
        try:
            text = html.unescape(text)
        except:
            pass
        
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Restore line breaks for key sections to maintain structure
        for section in ["NOMBRE DEL MEDICAMENTO", "QUÉ ES", "ANTES DE", "CÓMO TOMAR", 
                       "POSIBLES EFECTOS", "CONSERVACIÓN", "INFORMACIÓN ADICIONAL",
                       "1.", "2.", "3.", "4.", "5.", "6.", "7."]:
            text = text.replace(section, f"\n\n{section}")
        
        # Final cleanup
        return text.strip()

    async def generate_prospecto(self, query: str) -> Dict[str, str]:
        """
        Generate a complete medication prospecto based on CIMA data,
        following official AEMPS guidelines for patient information leaflets.
        
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
        
        # Get context for the medication - enhanced to prioritize prospecto content
        context = await self.get_medication_context(query)
        
        if "Error" in context or "No se encontraron" in context:
            return {
                "prospecto": f"No se pudo generar el prospecto: {context}",
                "context": context,
                "medication": query_info.get("active_principle", "")
            }
        
        # Create a prompt for generating the prospecto
        prompt = f"""
Por favor, genera un PROSPECTO DE MEDICAMENTO siguiendo estrictamente el formato oficial AEMPS.
El prospecto debe estar dirigido a PACIENTES (no a profesionales sanitarios) en lenguaje claro y accesible.

INFORMACIÓN DEL MEDICAMENTO Y CONTENIDO DEL PROSPECTO ORIGINAL:
{context}

CONSULTA DEL USUARIO:
{query}

Basándote en esta información, genera un prospecto completo que siga la estructura oficial AEMPS:

1. NOMBRE DEL MEDICAMENTO
2. QUÉ ES [MEDICAMENTO] Y PARA QUÉ SE UTILIZA
3. ANTES DE TOMAR [MEDICAMENTO]
4. CÓMO TOMAR [MEDICAMENTO]
5. POSIBLES EFECTOS ADVERSOS
6. CONSERVACIÓN DE [MEDICAMENTO]
7. INFORMACIÓN ADICIONAL

Utiliza un lenguaje claro, sencillo y directo. Evita tecnicismos innecesarios y explica los términos médicos cuando sean imprescindibles. Utiliza frases cortas y directas con formato de lista cuando sea apropiado para facilitar la comprensión.
"""

        try:
            # Generate the prospecto using OpenAI
            response = await self.openai_client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
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