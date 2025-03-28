"""
Prospecto generator module for creating medication package inserts.
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
    Generator for medication package inserts (prospectos) using CIMA data.
    """
    openai_client: AsyncOpenAI
    reference_cache: Dict[str, List[Dict]] = field(default_factory=dict)
    session: aiohttp.ClientSession = None
    max_tokens: int = 14000  # Leave room for prompt and response
    use_langgraph: bool = True  # Use the LangGraph search by default
    base_url: str = Config.CIMA_BASE_URL
    
    # System prompt specifically for prospecto generation
    system_prompt = """Experto en redacción de prospectos de medicamentos según normativa AEMPS.

Cuando se solicite redactar un prospecto completo, deberás generar un documento que cumpla con todos los requisitos oficiales de la AEMPS para prospectos de medicamentos, incluyendo:

1. DENOMINACIÓN DEL MEDICAMENTO
   - Nombre completo con forma farmacéutica y concentración

2. COMPOSICIÓN CUALITATIVA Y CUANTITATIVA
   - Principios activos y excipientes con declaración obligatoria

3. QUÉ ES Y PARA QUÉ SE UTILIZA
   - Descripción en lenguaje comprensible para el paciente
   - Grupo farmacoterapéutico
   - Indicaciones terapéuticas aprobadas

4. ANTES DE TOMAR EL MEDICAMENTO
   - Contraindicaciones
   - Advertencias y precauciones
   - Interacciones con otros medicamentos
   - Uso durante embarazo y lactancia
   - Efectos sobre la capacidad de conducir
   - Información importante sobre excipientes

5. CÓMO TOMAR EL MEDICAMENTO
   - Posología detallada para cada indicación
   - Forma de administración
   - Duración del tratamiento
   - Instrucciones en caso de sobredosis o dosis olvidadas

6. POSIBLES EFECTOS ADVERSOS
   - Clasificados por frecuencia según MedDRA
   - Agrupados por sistemas orgánicos
   - Instrucciones al paciente en caso de efectos adversos

7. CONSERVACIÓN
   - Condiciones específicas de temperatura y humedad
   - Periodo de validez
   - Precauciones especiales de conservación

8. INFORMACIÓN ADICIONAL
   - Titular de la autorización y responsable de fabricación
   - Fecha de última revisión

Utiliza un lenguaje claro, comprensible para el paciente medio sin conocimientos médicos. Estructura el texto con encabezados numerados. Evita terminología técnica innecesaria pero mantén la precisión científica.

Basa toda la información en los datos proporcionados en el contexto CIMA, citando apropiadamente las fuentes con el formato [Ref X: Nombre del medicamento (Nº Registro)].
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
        
        # Prospecto request detection pattern
        prospecto_pattern = r'(?:redactar|generar|crear|elaborar|realizar?e?|escrib[ei]r|hac[ae]r|desarroll[ae]r|realiza(?:r|)|prepar(?:ar|a))\s+(?:un|el|uns?|una?)?\s+prospecto'
        is_prospecto = bool(re.search(prospecto_pattern, query_lower))
        
        # Check if this mentions "prospecto" at all
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
                    words = [w for w in query_lower.split() if len(w) > 4 and not any(x in w for x in ['como', 'para', 'sobre', 'cual', 'este', 'esta'])]
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
        Gets medication context for prospecto generation using CIMA data.
        
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
        
        # Get the most relevant medication
        top_med = results[0]
        nregistro = top_med.get("nregistro")
        
        if not nregistro:
            return "No se pudo encontrar el número de registro del medicamento."
        
        # Get medication details
        try:
            session = await self.get_session()
            url = f"{self.base_url}/medicamento"
            
            async with session.get(url, params={"nregistro": nregistro}) as response:
                if response.status != 200:
                    return f"Error al obtener información del medicamento: {response.status}"
                    
                basic_info = await response.json()
            
            # Get prospecto content specifically
            prospecto_url = f"{self.base_url}/docSegmentado/contenido/2"
            
            async with session.get(prospecto_url, params={"nregistro": nregistro}) as response:
                if response.status != 200:
                    return f"Error al obtener el prospecto del medicamento: {response.status}"
                    
                prospecto_data = await response.json()
                prospecto_content = prospecto_data.get("contenido", "No disponible")
            
            # Format the context
            context = f"""
INFORMACIÓN DEL MEDICAMENTO:
- Nombre: {basic_info.get('nombre', 'No disponible')}
- Número de registro: {nregistro}
- Principio(s) activo(s): {basic_info.get('pactivos', 'No disponible')}
- Laboratorio titular: {basic_info.get('labtitular', 'No disponible')}

PROSPECTO ORIGINAL:
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

    async def generate_prospecto(self, query: str) -> Dict[str, str]:
        """
        Generate a complete medication prospecto based on CIMA data.
        
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
Por favor, genera un prospecto para medicamento siguiendo la normativa AEMPS basado en la siguiente información:

CONSULTA DEL USUARIO:
{query}

CONTEXT FROM CIMA:
{context}

El prospecto debe seguir la estructura oficial de la AEMPS, incluyendo todas las secciones requeridas y utilizando un lenguaje claro y comprensible para pacientes. Basa toda la información en los datos proporcionados en el contexto CIMA.
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