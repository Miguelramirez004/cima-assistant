from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Union
from openai import AsyncOpenAI
from config import Config
import aiohttp
import re
import json
import asyncio
from datetime import datetime
import logging
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FormulationAgent:
    openai_client: AsyncOpenAI
    base_url: str = Config.CIMA_BASE_URL
    reference_cache: Dict[str, List[Dict]] = field(default_factory=dict)
    session: aiohttp.ClientSession = None
    max_tokens: int = 14000  # Leave room for prompt and response

    system_prompt = """Farmacéutico especialista en formulación magistral con amplio conocimiento en CIMA. 
Genera formulaciones magistrales detalladas y precisas basadas en la información proporcionada por CIMA.

ESTRUCTURA DE RESPUESTA:

1. RESUMEN EJECUTIVO:
   - Breve descripción de la formulación y su finalidad terapéutica
   - Tipo de formulación (suspensión, solución, gel, pomada, etc.)
   - Concentración de principio(s) activo(s)

2. COMPOSICIÓN CUALITATIVA Y CUANTITATIVA:
   - Principio(s) activo(s): {nombre_compuesto} {concentración exacta}
   - Excipientes: Lista detallada con cantidades precisas
   - Justificación de la selección de excipientes

3. MATERIALES NECESARIOS:
   - Equipamiento específico requerido
   - Material de laboratorio necesario
   - Utillaje de precisión
   - EPIs recomendados

4. PROCEDIMIENTO DE ELABORACIÓN:
   - Métodos específicos para cada fase
   - Parámetros críticos (temperatura, pH, velocidad de agitación)
   - Orden preciso de incorporación de componentes
   - Precauciones especiales durante el proceso
   - Técnicas de homogeneización

5. ESPECIFICACIONES TÉCNICAS:
   - Características organolépticas
   - Parámetros físico-químicos (pH, viscosidad, densidad)
   - Criterios de conformidad farmacotécnica
   - Rango de valores aceptables

6. CONTROL DE CALIDAD:
   - Controles durante proceso de elaboración
   - Controles en producto terminado
   - Criterios de aceptación y rechazo
   - Documentación requerida

7. ENVASADO Y ACONDICIONAMIENTO:
   - Tipo de envase recomendado con justificación
   - Material de acondicionamiento
   - Condiciones de envasado

8. ESTABILIDAD Y CONSERVACIÓN:
   - Periodo de validez con justificación científica
   - Condiciones específicas de conservación
   - Signos de inestabilidad a vigilar
   - Estudios de estabilidad disponibles

9. ETIQUETADO:
   - Composición cualitativa y cuantitativa completa
   - Vía de administración y posología recomendada
   - Condiciones de conservación
   - Fecha límite de utilización
   - Advertencias y precauciones especiales
   - Instrucciones de uso para el paciente

10. INFORMACIÓN ADICIONAL:
    - Biodisponibilidad y consideraciones biofarmacéuticas
    - Monitorización específica recomendada
    - Alternativas terapéuticas
    - Interacciones relevantes a considerar

Para cada sección, incluye referencias específicas a las fuentes CIMA utilizadas, con el formato [Ref X: Nombre del medicamento (Nº Registro)]. Utiliza la información proporcionada en el contexto para justificar tus decisiones y especificar cantidades exactas.

Si hay información insuficiente para alguna sección, indícalo claramente y sugiere fuentes adicionales que podrían consultarse.
"""

    prospecto_prompt = """Experto en redacción de prospectos de medicamentos según normativa AEMPS.

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
        self.openai_client = openai_client
        self.reference_cache = {}
        self.base_url = Config.CIMA_BASE_URL
        self.session = None
        self.max_tokens = 14000
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
            "vitamina d", "calcio", "hierro", "insulina", "metronidazol"
        ]

    def num_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a string"""
        return len(self.tokenizer.encode(text))

    async def get_session(self):
        """Get or create an aiohttp session with keepalive"""
        if self.session is None or self.session.closed:
            # Using TCPConnector with proper settings
            connector = aiohttp.TCPConnector(
                ssl=False,  
                limit=10,  # Reduced connection limit to prevent overload
                keepalive_timeout=60,  # Shorter keepalive period
                force_close=False  # Don't force close connections
            )
            timeout = aiohttp.ClientTimeout(
                total=30,  # Shorter timeout
                connect=10,
                sock_connect=10,
                sock_read=10
            )
            self.session = aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout,
                raise_for_status=False  # Don't raise exceptions for HTTP errors
            )
        return self.session
    
    async def get_medication_details(self, nregistro: str) -> Dict:
        """Optimized method to fetch medication details concurrently"""
        logger.info(f"Fetching medication details for nregistro: {nregistro}")
        session = await self.get_session()
        
        # Most important sections first, so if we need to truncate, we keep the important ones
        sections_of_interest = {
            "2": "composicion",
            "4.1": "indicaciones",
            "4.2": "posologia_procedimiento",
            "4.3": "contraindicaciones",
            "4.4": "advertencias",
            "4.5": "interacciones",
            "6.1": "excipientes",
            "6.3": "conservacion",
            # Less critical sections after
            "4.6": "embarazo_lactancia",
            "4.8": "efectos_adversos",
            "5.1": "propiedades_farmacodinamicas",
            "5.2": "propiedades_farmacocineticas",
            "5.3": "datos_preclinicos",
            "6.2": "incompatibilidades",
            "6.4": "especificaciones",
            "6.5": "envase",
            "6.6": "eliminacion",
            "7": "titular_autorizacion",
            "8": "numero_autorizacion",
            "9": "fecha_autorizacion",
            "10": "fecha_revision"
        }
        
        details = {}
        
        # Define async tasks for concurrent execution
        async def get_basic_info():
            detail_url = f"{self.base_url}/medicamento"
            try:
                async with session.get(detail_url, params={"nregistro": nregistro}) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, dict):
                            return result
            except Exception as e:
                logger.error(f"Error retrieving basic details: {str(e)}")
            return {"error": "Unable to retrieve basic details"}
        
        async def get_section(section, key):
            tech_url = f"{self.base_url}/docSegmentado/contenido/1"
            params = {"nregistro": nregistro, "seccion": section}
            
            # First try
            try:
                async with session.get(tech_url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, dict):
                            if result.get("contenido") and result.get("contenido") != "No disponible":
                                return (key, result)
            except Exception as e:
                logger.warning(f"Error in first attempt for section {section}: {str(e)}")
            
            # Second try - specific URL for technical sheet section
            try:
                tech_url_html = f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/{section}/FichaTecnica.html"
                async with session.get(tech_url_html) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        if html_content and len(html_content) > 50:  # Simple validity check
                            return (key, {"contenido": html_content})
            except Exception as e:
                logger.warning(f"Error in second attempt for section {section}: {str(e)}")
            
            # Fallback to full technical sheet and extract section
            try:
                full_tech_url = f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html"
                async with session.get(full_tech_url) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        if html_content and len(html_content) > 100:
                            # Try to extract the section
                            section_pattern = f'<h3[^>]*>{section}\\.[^<]+</h3>(.*?)<h3'
                            match = re.search(section_pattern, html_content, re.DOTALL)
                            if match:
                                return (key, {"contenido": match.group(1)})
            except Exception as e:
                logger.warning(f"Error in fallback attempt for section {section}: {str(e)}")
            
            return (key, {"contenido": f"No disponible - Consultar ficha técnica completa"})
        
        async def get_prospecto_section(section=None):
            """Get prospecto content with improved error handling and multiple methods"""
            # First approach - JSON API
            url = f"{self.base_url}/docSegmentado/contenido/2"
            params = {"nregistro": nregistro}
            if section:
                params["seccion"] = section
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        try:
                            # Try to parse as JSON first
                            result = await response.json()
                            if isinstance(result, dict) and "contenido" in result:
                                if result.get("contenido") and result.get("contenido") != "No disponible":
                                    return {"prospecto_html": result.get("contenido", "")}
                        except:
                            # If not JSON, try as text
                            content = await response.text()
                            if content and len(content) > 50:
                                return {"prospecto_html": content}
            except Exception as e:
                logger.warning(f"Error in first prospecto retrieval attempt: {str(e)}")
            
            # Second approach - HTML API with specific header
            try:
                headers = {"Accept": "text/html"}
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        if content and len(content) > 50:
                            return {"prospecto_html": content}
            except Exception as e:
                logger.warning(f"Error in second prospecto retrieval attempt: {str(e)}")
            
            # Third approach - Direct HTML access
            try:
                # If section is specified, try section-specific URL
                if section:
                    html_url = f"https://cima.aemps.es/cima/dochtml/p/{nregistro}/{section}/Prospecto.html"
                else:
                    html_url = f"https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html"
                
                async with session.get(html_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        if content and len(content) > 50:
                            return {"prospecto_html": content}
            except Exception as e:
                logger.warning(f"Error in direct HTML prospecto retrieval: {str(e)}")
            
            # Fallback - just provide link
            return {"prospecto_html": f"Prospecto disponible en: https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html"}
        
        # Execute basic info request first - we need this for sure
        basic_info = await get_basic_info()
        details["basic"] = basic_info
        
        # Only fetch sections if we got basic info successfully
        if "error" not in basic_info:
            # Limit concurrent section requests to avoid overwhelming the API
            semaphore = asyncio.Semaphore(5)  # Reduced from 10 to 5 concurrent requests
            
            async def limited_section_fetch(section, key):
                async with semaphore:
                    return await get_section(section, key)
            
            # Create tasks for each section (with concurrency limit)
            section_tasks = [limited_section_fetch(section, key) for section, key in sections_of_interest.items()]
            prospecto_task = get_prospecto_section()
            
            # Execute section tasks concurrently and process results
            section_results = await asyncio.gather(*section_tasks)
            prospecto_result = await prospecto_task
            
            for key, value in section_results:
                details[key] = value
                
            details["prospecto"] = prospecto_result
        
        logger.info(f"Completed fetching details for nregistro: {nregistro}")
        return details

    def detect_formulation_type(self, query: str) -> Dict[str, Any]:
        """
        Enhanced formulation type detection with additional parameters
        """
        # Dictionary for formulation types and their keywords
        formulation_types = Config.FORMULATION_TYPES
        
        # Dictionary for pharmaceutical paths
        admin_routes = Config.ADMIN_ROUTES
        
        # Concentration patterns
        concentration_pattern = r'(\d+(?:[,.]\d+)?)\s*(%|mg|g|ml|mcg|UI|unidades)'
        
        # Special request patterns - Enhanced to catch all variants
        prospecto_pattern = r'(?:redactar|generar|crear|elaborar|realizar?e?|escrib[ei]r|hac[ae]r|desarroll[ae]r|realiza(?:r|)|prepar(?:ar|a))\s+(?:un|el|uns?|una?)?\s+prospecto'
        
        # Process query
        query_lower = query.lower()
        
        # Detect if this is a special request for a prospecto
        is_prospecto = bool(re.search(prospecto_pattern, query_lower))
        logger.info(f"Query: '{query}', Is prospecto: {is_prospecto}")
        
        # Detect formulation type
        detected_form = "suspension"  # Default
        for form_type, keywords in formulation_types.items():
            if any(word in query_lower for word in keywords):
                detected_form = form_type
                break
        
        # Detect administration route
        detected_route = "oral"  # Default
        for route, keywords in admin_routes.items():
            if any(word in query_lower for word in keywords):
                detected_route = route
                break
        
        # Extract concentration if present
        concentration_match = re.search(concentration_pattern, query)
        concentration = concentration_match.group(0) if concentration_match else None
        
        # Extract active principles from our database
        active_principle = None
        for ap in self.active_principles:
            if ap in query_lower:
                active_principle = ap
                break
        
        # If not found, try to extract from capitalization
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
        
        return {
            "form_type": detected_form,
            "admin_route": detected_route,
            "concentration": concentration,
            "active_principle": active_principle,
            "is_prospecto": is_prospecto
        }

    def format_medication_info(self, index: int, med: Dict, details: Dict) -> str:
        """
        Streamlined medication information formatting to reduce token usage
        """
        # Basic medication info - with safe access
        basic_info = details.get('basic', {})
        if not isinstance(basic_info, dict):
            basic_info = {}
            
        nregistro = med.get('nregistro', 'No disponible')
        med_name = med.get('nombre', 'No disponible')
        
        # Format date if available
        fecha_autorizacion = basic_info.get('fechaAutorizacion', '')
        if fecha_autorizacion:
            try:
                fecha_obj = datetime.strptime(fecha_autorizacion, "%Y%m%d")
                fecha_autorizacion = fecha_obj.strftime("%d/%m/%Y")
            except:
                pass
        
        # Get laboratory information
        lab_titular = basic_info.get('labtitular', 'No disponible')
        
        # Format sections with proper handling of missing data and truncation
        def get_section_content(section_key, max_len=500):  # Increased max length
            section_data = details.get(section_key, {})
            if not isinstance(section_data, dict):
                return "No disponible"
            
            content = section_data.get('contenido', 'No disponible')
            # Only truncate if really necessary and include indication this is truncated
            if content and len(content) > max_len:
                return content[:max_len] + "... [Contenido truncado, ver ficha técnica completa]"
            return content
        
        # Include most important sections first
        reference = f"""
[Referencia {index}: {med_name} (Nº Registro: {nregistro})]

INFORMACIÓN BÁSICA:
- Nombre: {med_name}
- Número de registro: {nregistro}
- Laboratorio titular: {lab_titular}
- Principios activos: {med.get('pactivos', 'No disponible')}

COMPOSICIÓN:
{get_section_content('composicion')}

INDICACIONES TERAPÉUTICAS:
{get_section_content('indicaciones')}

POSOLOGÍA Y ADMINISTRACIÓN:
{get_section_content('posologia_procedimiento')}

CONTRAINDICACIONES:
{get_section_content('contraindicaciones')}

ADVERTENCIAS Y PRECAUCIONES:
{get_section_content('advertencias')}

EXCIPIENTES:
{get_section_content('excipientes')}

CONSERVACIÓN:
{get_section_content('conservacion')}
"""

        # Add URLs
        reference += f"""
URL FICHA TÉCNICA:
https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html

URL PROSPECTO:
https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html
"""
        return reference

    async def get_relevant_context(self, query: str, n_results: int = 3) -> str:
        """
        Enhanced context retrieval with improved search strategies
        """
        logger.info(f"Getting context for query: '{query}'")
        cache_key = f"{query}_{n_results}"
        
        # Enhanced formulation detection
        formulation_info = self.detect_formulation_type(query)
        formulation_type = formulation_info["form_type"]
        active_principle = formulation_info["active_principle"]
        is_prospecto = formulation_info["is_prospecto"]
        
        # Extract medication name from prospecto request
        if is_prospecto:
            # Try to extract medication name using patterns
            med_patterns = [
                r'(?:sobre|de|para|con)\s+(?:la|el|los|las)?\s*([a-zA-Z\-áéíóúÁÉÍÓÚñÑ]+(?:\s+[a-zA-Z\-áéíóúÁÉÍÓÚñÑ]+){0,3})\s+(\d+(?:[,.]\d+)?\s*(?:mg|g|ml|mcg|UI|%|unidades))',  # With concentration
                r'(?:sobre|de|para|con)\s+(?:la|el|los|las)?\s*([a-zA-Z\-áéíóúÁÉÍÓÚñÑ]+(?:\s+[a-zA-Z\-áéíóúÁÉÍÓÚñÑ]+){0,3})',  # Without concentration
                r'prospecto\s+(?:sobre|de|para|con)?\s+(?:la|el|los|las)?\s*([a-zA-Z\-áéíóúÁÉÍÓÚñÑ]+(?:\s+[a-zA-Z\-áéíóúÁÉÍÓÚñÑ]+){0,3})'  # After prospecto
            ]
            
            for pattern in med_patterns:
                match = re.search(pattern, query.lower())
                if match:
                    extracted_term = match.group(1).strip()
                    if len(extracted_term) > 3:  # Avoid short/common words
                        active_principle = extracted_term
                        logger.info(f"Extracted medication term from prospecto request: '{active_principle}'")
                        break
        
        if cache_key in self.reference_cache:
            cached_results = self.reference_cache[cache_key]
            context_parts = [self.format_medication_info(i, med, details) 
                            for i, (med, details) in enumerate(cached_results, 1)]
            
            # Calculate token count and truncate if needed
            full_context = "\n".join(context_parts)
            if self.num_tokens(full_context) > self.max_tokens:
                logger.info(f"Context too large ({self.num_tokens(full_context)} tokens), truncating...")
                # Use fewer results
                smaller_context = "\n".join(context_parts[:2])
                if self.num_tokens(smaller_context) > self.max_tokens:
                    # Truncate even more - just use most relevant result
                    return context_parts[0]
                return smaller_context
            return full_context

        # Get/create aiohttp session
        session = await self.get_session()
        
        # Execute search with multiple strategies
        all_results = await self._comprehensive_medication_search(session, query, active_principle, formulation_type)
        
        # Limit results to requested number
        results = all_results[:n_results]
        cached_results = []

        if results:
            logger.info(f"Found {len(results)} relevant medications, fetching details")
            
            # Fetch details for all medications concurrently
            semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
            
            async def limited_fetch_med_details(med):
                async with semaphore:
                    if not isinstance(med, dict) or not med.get("nregistro"):
                        return None
                    try:
                        details = await self.get_medication_details(med["nregistro"])
                        return (med, details)
                    except Exception as e:
                        logger.error(f"Error fetching details for {med.get('nregistro', 'unknown')}: {str(e)}")
                        return None
            
            # Create tasks for fetching medication details
            detail_tasks = [limited_fetch_med_details(med) for med in results]
            detail_results = await asyncio.gather(*detail_tasks)
            
            # Filter out None results
            cached_results = [result for result in detail_results if result is not None]
            
            # Store in cache
            self.reference_cache[cache_key] = cached_results
            
            # Format context with complete information
            context_parts = [self.format_medication_info(i, med, details) 
                          for i, (med, details) in enumerate(cached_results, 1)]
            
            # Calculate token count and truncate if needed
            full_context = "\n".join(context_parts)
            if self.num_tokens(full_context) > self.max_tokens:
                logger.info(f"Context too large ({self.num_tokens(full_context)} tokens), truncating...")
                # Use fewer results
                smaller_context = "\n".join(context_parts[:2])
                if self.num_tokens(smaller_context) > self.max_tokens:
                    # Truncate even more - just use most relevant result
                    return context_parts[0]
                return smaller_context
            
            return full_context
        
        return "No se encontraron medicamentos relevantes en CIMA para esta consulta. Por favor intente con términos más específicos o verifique que el principio activo o medicamento esté registrado en CIMA."

    async def _comprehensive_medication_search(self, session, query, active_principle=None, form_type=None):
        """
        Comprehensive search strategy using multiple approaches to find relevant medications
        """
        all_results = []
        seen_nregistros = set()
        search_url = f"{self.base_url}/medicamentos"
        
        # Extract all potential search terms
        search_terms = self._extract_search_terms(query)
        
        # Helper function for search
        async def execute_search(params, description):
            results = []
            try:
                logger.info(f"Executing {description} search with params: {params}")
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, dict) and "resultados" in data:
                            results = data.get("resultados", [])
                            logger.info(f"{description} search returned {len(results)} results")
            except Exception as e:
                logger.error(f"Error in {description} search: {str(e)}")
            return results
        
        # 1. Search by exact active principle
        if active_principle:
            results = await execute_search({"principiosActivos": active_principle}, "active principle")
            self._add_unique_results(results, all_results, seen_nregistros)
            
            # If no results, try variations
            if not all_results and len(active_principle) > 4:
                # Try without accents and other variations
                variations = [
                    active_principle.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u'),
                    active_principle.replace('a', 'á').replace('e', 'é').replace('i', 'í').replace('o', 'ó').replace('u', 'ú'),
                    active_principle.lower(),
                    active_principle.upper(),
                    active_principle.capitalize()
                ]
                
                for var in variations:
                    if var != active_principle:
                        results = await execute_search({"principiosActivos": var}, f"active principle variation '{var}'")
                        if self._add_unique_results(results, all_results, seen_nregistros) and len(all_results) >= 3:
                            break
        
        # 2. Search by name with exact query
        if len(all_results) < 3:
            results = await execute_search({"nombre": query}, "name with full query")
            self._add_unique_results(results, all_results, seen_nregistros)
        
        # 3. Search by individual search terms
        if len(all_results) < 3 and search_terms:
            for term in search_terms:
                if len(all_results) >= 5:
                    break
                    
                # Skip terms that are too common or irrelevant
                if len(term) < 4 or term.lower() in ['para', 'como', 'sobre', 'este', 'esta', 'estos']:
                    continue
                    
                results = await execute_search({"nombre": term}, f"term '{term}'")
                self._add_unique_results(results, all_results, seen_nregistros)
                
                # If this term gave good results, prioritize it
                if len(results) > 0:
                    # Try active principle search with this term
                    pa_results = await execute_search({"practiv1": term}, f"practiv1 with term '{term}'")
                    self._add_unique_results(pa_results, all_results, seen_nregistros)
        
        # 4. Search by form type if specified
        if len(all_results) < 3 and form_type:
            results = await execute_search({"formaFarmaceutica": form_type}, "form type")
            
            # For form type search, add a filter to ensure relevance
            filtered_results = []
            for med in results:
                # Only add if it might be relevant
                if (active_principle and active_principle.lower() in med.get('nombre', '').lower()) or \
                   (active_principle and active_principle.lower() in med.get('pactivos', '').lower()) or \
                   any(term.lower() in med.get('nombre', '').lower() for term in search_terms):
                    filtered_results.append(med)
                    
            self._add_unique_results(filtered_results, all_results, seen_nregistros)
        
        # 5. As a fallback for specific important medications that might not be in CIMA
        if not all_results and active_principle and active_principle.lower() == "melatonina":
            # Create a synthetic entry with basic info for melatonina
            melatonina_info = {
                "nregistro": "custom_melatonina",
                "nombre": "Melatonina (Suplemento Dietético)",
                "pactivos": "Melatonina",
                "labtitular": "Varios fabricantes",
                "custom_info": {
                    "descripcion": "La melatonina es una hormona producida por la glándula pineal que regula el ciclo de sueño-vigilia. Como suplemento dietético, se utiliza para trastornos del sueño y jet lag.",
                    "indicaciones": "Insomnio, trastornos del sueño, jet lag, regulación del ciclo circadiano.",
                    "posologia": "La dosis habitual oscila entre 0.5 y 5 mg tomados 30-60 minutos antes de acostarse.",
                    "contraindicaciones": "Embarazo, lactancia, enfermedades autoinmunes, depresión. Consultar con su médico antes de usar.",
                    "efectos_adversos": "Somnolencia, dolor de cabeza, mareos, náuseas, irritabilidad."
                }
            }
            all_results.append(melatonina_info)
        
        # Ensure we have most relevant results first by sorting
        if active_principle:
            def relevance_score(med):
                name = med.get('nombre', '').lower()
                pactivos = med.get('pactivos', '').lower()
                # Higher score means more relevant
                score = 0
                
                # Check if active principle matches exactly
                if active_principle.lower() in pactivos:
                    score += 100
                    
                # Check if active principle is in the name
                if active_principle.lower() in name:
                    score += 50
                    
                # Check for concentration match if specified in query
                concentration_match = re.search(r'(\d+(?:[,.]\d+)?)\s*(%|mg|g|ml|mcg|UI|unidades)', query.lower())
                if concentration_match and concentration_match.group(0) in name:
                    score += 30
                
                # Prioritize comercialized products
                if med.get('comerc', False):
                    score += 20
                    
                return score
                
            all_results.sort(key=relevance_score, reverse=True)
        
        return all_results

    def _add_unique_results(self, new_results, all_results, seen_nregistros):
        """Add unique results to the all_results list and update seen_nregistros set"""
        added = False
        for med in new_results:
            if isinstance(med, dict) and med.get("nregistro"):
                nregistro = med.get("nregistro")
                if nregistro not in seen_nregistros:
                    seen_nregistros.add(nregistro)
                    all_results.append(med)
                    added = True
        return added

    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract potential search terms from the query"""
        # Patterns for potential medication names and active ingredients
        patterns = [
            r'([A-Z][a-záéíóúñ]+(?:\s[a-záéíóúñ]+){0,3})',  # Capitalized words
            r'(\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|UI|unidades))',  # Dosages
            r'([A-Za-záéíóúñ]+\+[A-Za-záéíóúñ]+)'  # Combinations with +
        ]
        
        # Extract terms using all patterns
        potential_terms = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            potential_terms.extend([m.strip() for m in matches if len(m.strip()) > 3])
        
        # Add individual words that might be medication names
        common_words = {"sobre", "para", "como", "este", "esta", "estos", "estas", "cual", "cuales", 
                       "con", "por", "los", "las", "del", "que", "realizar", "realizar", "redactar", 
                       "crear", "generar", "prospecto", "formular", "elaborar", "realiza", "prepara"}
        
        words = query.split()
        for word in words:
            if len(word) > 4 and word.lower() not in common_words and word not in potential_terms:
                potential_terms.append(word)
        
        # Add bi-grams (pairs of words)
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i+1]) > 3:
                bigram = f"{words[i]} {words[i+1]}"
                if bigram not in potential_terms:
                    potential_terms.append(bigram)
        
        # Check for active principles in our database
        for ap in self.active_principles:
            if ap in query.lower() and ap not in potential_terms:
                potential_terms.append(ap)
        
        # Eliminate duplicates
        return list(set(potential_terms))

    async def generate_response(self, query: str, context: str) -> str:
        """
        Generate response with selected system prompt based on query type
        """
        # Extract formulation details for improved prompting
        formulation_info = self.detect_formulation_type(query)
        
        # Select the appropriate system prompt based on query type
        system_prompt = self.prospecto_prompt if formulation_info["is_prospecto"] else self.system_prompt
        
        # Count tokens before creating prompt
        system_tokens = self.num_tokens(system_prompt)
        query_tokens = self.num_tokens(query)
        context_tokens = self.num_tokens(context)
        
        logger.info(f"Token counts - System: {system_tokens}, Query: {query_tokens}, Context: {context_tokens}")
        
        # Create prompt with complete context
        prompt = f"""
Analiza el siguiente contexto para generar una respuesta detallada:

CONTEXTO:
{context}

DETALLES DE LA CONSULTA:
- Tipo de formulación: {formulation_info["form_type"]}
- Vía de administración: {formulation_info["admin_route"]}
- Principio(s) activo(s): {formulation_info["active_principle"]}
- Concentración solicitada: {formulation_info["concentration"] if formulation_info["concentration"] else "No especificada"}
- Es solicitud de prospecto: {"Sí" if formulation_info["is_prospecto"] else "No"}

CONSULTA ORIGINAL:
{query}

Genera una respuesta completa y exhaustiva utilizando toda la información disponible en el contexto. Cita las fuentes CIMA usando el formato [Ref X: Nombre del medicamento (Nº Registro)].

Si no encuentras información suficiente en el contexto, indícalo claramente y sugiere consultar directamente la ficha técnica a través de los enlaces proporcionados.
"""

        prompt_tokens = self.num_tokens(prompt)
        total_tokens = system_tokens + prompt_tokens
        
        logger.info(f"Total input tokens: {total_tokens}")
        
        if total_tokens > 15000:
            logger.warning(f"Token count {total_tokens} approaching limit, might encounter issues")
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {str(e)}")
            raise Exception(f"Error al generar la respuesta: {str(e)}")

    async def answer_question(self, question: str) -> Dict[str, str]:
        context = await self.get_relevant_context(question)
        answer = await self.generate_response(question, context)
        
        # Create direct links for references in response
        pattern = r'\[Ref (\d+): ([^()]+) \(Nº Registro: (\d+)\)\]'
        
        def replace_with_link(match):
            ref_num = match.group(1)
            med_name = match.group(2)
            reg_num = match.group(3)
            return f'[Ref {ref_num}: {med_name} (Nº Registro: {reg_num})](https://cima.aemps.es/cima/dochtml/ft/{reg_num}/FT_{reg_num}.html)'
        
        answer_with_links = re.sub(pattern, replace_with_link, answer)
        
        # Handle custom melatonina case specially
        if "melatonina" in question.lower() and "custom_melatonina" in context:
            answer_with_links = answer_with_links.replace("[Ref: Melatonina (Suplemento Dietético) (Nº Registro: custom_melatonina)]", 
                                                         "**Nota sobre Melatonina:** La melatonina suele comercializarse como suplemento alimenticio, no como medicamento registrado en CIMA")
        
        return {
            "answer": answer_with_links,
            "context": context,
            "references": context.count("[Referencia")
        }
    
    async def close(self):
        """Close the aiohttp session to free resources with proper error handling"""
        if self.session and not self.session.closed:
            try:
                await self.session.close()
                # Give the event loop time to clean up connections
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error closing session: {str(e)}")
                # Ensure session is marked as closed even if there was an error
                self.session = None

@dataclass
class CIMAExpertAgent:
    openai_client: AsyncOpenAI
    reference_cache: Dict[str, str] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    base_url: str = Config.CIMA_BASE_URL
    session: aiohttp.ClientSession = None
    max_tokens: int = 14000  # Reserve tokens for prompt and response
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.reference_cache = {}
        self.base_url = Config.CIMA_BASE_URL
        self.conversation_history = []
        self.session = None
        self.max_tokens = 14000
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
            "vitamina d", "calcio", "hierro", "insulina", "metronidazol"
        ]

    def num_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a string"""
        return len(self.tokenizer.encode(text))

    system_prompt = """Experto conversacional en medicamentos y CIMA (Centro de Información online de Medicamentos de la AEMPS). 

Mantén un diálogo natural, educativo y coherente, recordando el contexto previo de la conversación.

Para cada respuesta:
1. Utiliza SIEMPRE la información proporcionada en el contexto CIMA
2. Cita las fuentes específicas usando el formato [Ref: Nombre del medicamento (Nº Registro)]
3. Incluye enlaces directos a las fichas técnicas cuando sea relevante
4. Estructura tus respuestas con encabezados cuando sea apropiado
5. Indica claramente cuando la información no esté disponible en el contexto

Si necesitas más información para dar una respuesta completa:
- Haz preguntas específicas de seguimiento
- Solicita detalles concretos sobre el medicamento o la consulta
- Indica qué información adicional sería útil

Nunca inventes información que no aparezca en el contexto CIMA. Si no tienes suficiente información, indica qué datos específicos faltan y sugiere cómo el usuario podría completarlos.

Intenta explicar la información técnica en términos comprensibles sin perder precisión científica. Cuando sea necesario, define términos médicos o farmacéuticos complejos.

Recuerda que tus respuestas pueden tener impacto en decisiones de salud, así que mantén el rigor científico y la precisión en todo momento.
"""

    prospecto_prompt = """Experto en redacción de prospectos de medicamentos según normativa AEMPS.

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

Basa toda la información en los datos proporcionados en el contexto CIMA, citando apropiadamente las fuentes con el formato [Ref: Nombre del medicamento (Nº Registro)].

En caso de que no encuentres suficiente información en el contexto proporcionado, puedes basarte en información general conocida sobre el principio activo, pero indícalo claramente. Para principios activos comunes como melatonina que suelen venderse como suplementos alimenticios, adapta la información indicando claramente esta particularidad.
"""

    async def get_session(self):
        """Get or create an aiohttp session with keepalive"""
        if self.session is None or self.session.closed:
            # Using TCPConnector with proper settings
            connector = aiohttp.TCPConnector(
                ssl=False,  
                limit=10,  # Reduced connection limit to prevent overload
                keepalive_timeout=60,  # Shorter keepalive period
                force_close=False  # Don't force close connections
            )
            timeout = aiohttp.ClientTimeout(
                total=30,  # Shorter timeout
                connect=10,
                sock_connect=10,
                sock_read=10
            )
            self.session = aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout,
                raise_for_status=False  # Don't raise exceptions for HTTP errors
            )
        return self.session

    async def get_medication_info(self, query: str) -> str:
        """
        Improved implementation for obtaining medication information with better search capability
        """
        cache_key = f"query_{query}"
        if cache_key in self.reference_cache:
            return self.reference_cache[cache_key]

        # Enhanced query analysis
        query_info = self._analyze_query(query)
        logger.info(f"Query analysis: {query_info}")
        
        # Get the session
        session = await self.get_session()
        
        # Execute search with multiple strategies
        all_results = await self._comprehensive_medication_search(
            session, 
            query, 
            query_info["active_ingredient"], 
            query_info["is_prospecto"]
        )
        
        # Prepare medication information
        all_med_info = []
        processed_nregistros = set()
        
        # Process each medication
        for med in all_results[:5]:  # Limit to top 5 medications
            if med.get("nregistro") not in processed_nregistros:
                processed_nregistros.add(med.get("nregistro"))
                med_info = await self._get_complete_medication_details(
                    session, 
                    med, 
                    fetch_prospecto=query_info["is_prospecto"],
                    focus_section=query_info["section_number"]
                )
                if med_info:
                    all_med_info.append(med_info)
        
        # Special handling for unregistered medications
        if query_info["active_ingredient"] == "melatonina" and not all_med_info:
            all_med_info.append(self._create_melatonina_info())
        
        # Combine all results and check token count
        combined_results = "\n\n".join(all_med_info)
        
        # If the combined results exceed our token limit, truncate
        if self.num_tokens(combined_results) > self.max_tokens:
            logger.info(f"Context too large ({self.num_tokens(combined_results)} tokens), truncating...")
            # Try with fewer results first
            smaller_context = "\n\n".join(all_med_info[:2])
            if self.num_tokens(smaller_context) > self.max_tokens:
                # If still too large, just use the most relevant result
                smaller_context = all_med_info[0]
            combined_results = smaller_context
        
        # If we got no results, provide a helpful message
        if not all_med_info:
            combined_results = "No se encontraron resultados específicos para esta consulta en la base de datos CIMA. " + \
                              "Por favor, intente con nombres de medicamentos más específicos o reformule su consulta."
        
        # Store in cache
        self.reference_cache[cache_key] = combined_results
        return combined_results

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Enhanced query analysis to extract key information
        """
        query_lower = query.lower()
        is_prospecto = bool(re.search(r'(?:prospecto|reda[ckt]|crear?|generar?|prepar(?:ar?|))\s+(?:un|el|de)', query_lower))
        
        # Define patterns for important pharmaceutical terms
        section_patterns = {
            "4.1": [r'indicac', r'uso', r'para que', r'utiliza'],
            "4.2": [r'poso', r'dosis', r'como\s+(?:tomar|usar)', r'administra'],
            "4.3": [r'contrain', r'no\s+(?:debe|usar|tomar)', r'cuando\s+no'],
            "4.4": [r'advert', r'precau', r'vigil'],
            "4.5": [r'interac', r'con\s+otros', r'combina', r'junto'],
            "4.6": [r'embara', r'lactan', r'matern', r'ferti'],
            "4.8": [r'advers', r'efecto', r'secundario', r'reaccion', r'tolera'],
            "5.1": [r'farmaco.?dinam', r'mecanismo.?(?:accion|efecto)'],
            "5.2": [r'farmaco.?cinet', r'absorcion', r'metaboli', r'elimina']
        }
        
        # Find the appropriate section for the query
        section_number = None
        for section, patterns in section_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                section_number = section
                break
                
        # Try to find an active ingredient in our database
        active_ingredient = None
        for ap in self.active_principles:
            if ap in query_lower:
                active_ingredient = ap
                break
        
        # Extract all potential search terms
        search_terms = self._extract_search_terms(query)
        
        # If active ingredient wasn't found through our database, try to extract it from search terms
        if not active_ingredient and search_terms:
            # Use the first search term as a potential medication name
            active_ingredient = search_terms[0]
        
        return {
            "active_ingredient": active_ingredient,
            "section_number": section_number,
            "is_prospecto": is_prospecto,
            "search_terms": search_terms
        }

    async def _comprehensive_medication_search(self, session, query, active_principle=None, is_prospecto=False):
        """
        Enhanced medication search with multiple strategies and fallbacks
        """
        all_results = []
        seen_nregistros = set()
        search_url = f"{self.base_url}/medicamentos"
        
        # Extract all potential search terms
        search_terms = self._extract_search_terms(query)
        
        # Helper function for search
        async def execute_search(params, description):
            results = []
            try:
                logger.info(f"Executing {description} search with params: {params}")
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, dict) and "resultados" in data:
                            results = data.get("resultados", [])
                            logger.info(f"{description} search returned {len(results)} results")
            except Exception as e:
                logger.error(f"Error in {description} search: {str(e)}")
            return results
        
        # 1. Active principle search
        if active_principle:
            # Try exact match first
            results = await execute_search({"principiosActivos": active_principle}, "active principle")
            self._add_unique_results(results, all_results, seen_nregistros)
            
            # Try variations if needed
            if not all_results and len(active_principle) > 4:
                variations = [
                    active_principle.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u'),
                    active_principle.lower(),
                    active_principle.capitalize()
                ]
                
                for var in variations:
                    if var != active_principle:
                        results = await execute_search({"principiosActivos": var}, f"active principle variation '{var}'")
                        if self._add_unique_results(results, all_results, seen_nregistros) and len(all_results) >= 3:
                            break
            
            # Try practiv1 parameter if needed
            if len(all_results) < 2:
                results = await execute_search({"practiv1": active_principle}, "practiv1")
                self._add_unique_results(results, all_results, seen_nregistros)
        
        # 2. Full name search
        if len(all_results) < 3:
            results = await execute_search({"nombre": query}, "full name")
            self._add_unique_results(results, all_results, seen_nregistros)
        
        # 3. Individual search terms
        if len(all_results) < 3:
            for term in search_terms:
                if len(term) < 4 or term.lower() in ['para', 'como', 'sobre', 'este', 'esta', 'estos']:
                    continue
                    
                results = await execute_search({"nombre": term}, f"term '{term}'")
                self._add_unique_results(results, all_results, seen_nregistros)
                
                if len(all_results) >= 5:
                    break
        
        # Ensure relevant results are sorted to the top
        if active_principle and all_results:
            # Calculate a relevance score for each result
            def relevance_score(med):
                name = med.get('nombre', '').lower()
                pactivos = med.get('pactivos', '').lower()
                score = 0
                
                # Active principle match
                if active_principle.lower() in pactivos:
                    score += 100
                if active_principle.lower() in name:
                    score += 50
                    
                # Exact active principle
                if active_principle.lower() == pactivos.lower():
                    score += 200
                    
                # Check if any numbers in query match concentration in name
                numbers_in_query = re.findall(r'(\d+(?:[,.]\d+)?)\s*(?:%|mg|g|ml|mcg|UI|unidades)', query.lower())
                for num in numbers_in_query:
                    if num in name:
                        score += 30
                
                # Commercial vs generic
                if 'efg' in name.lower() or 'generico' in name.lower():
                    score -= 20  # Prefer brand names slightly
                    
                # Ensure ABACAVIR/LAMIVUDINA does not show up for unrelated searches
                if 'abacavir' in name.lower() and active_principle.lower() not in ['abacavir', 'lamivudina']:
                    score -= 1000  # Heavily penalize ABACAVIR showing up in wrong searches
                
                return score
                
            all_results.sort(key=relevance_score, reverse=True)
            
            # Double check to filter out obviously wrong medications
            if len(all_results) > 0 and active_principle and active_principle.lower() not in ['abacavir', 'lamivudina']:
                filtered_results = [med for med in all_results 
                                  if 'abacavir' not in med.get('nombre', '').lower() or 
                                     'abacavir' in active_principle.lower()]
                
                # Only use filtered results if we didn't filter out everything
                if filtered_results:
                    all_results = filtered_results
        
        # Special case for melatonina
        if active_principle and active_principle.lower() == "melatonina" and not all_results:
            melatonina_info = self._create_custom_medication_entry("melatonina", "Melatonina", "suplemento alimenticio")
            all_results.append(melatonina_info)
        
        return all_results

    def _create_custom_medication_entry(self, key, name, description):
        """Create a custom entry for medications not in CIMA but commonly searched"""
        return {
            "nregistro": f"custom_{key}",
            "nombre": f"{name} ({description})",
            "pactivos": name,
            "labtitular": "Varios fabricantes",
            "custom_info": True
        }

    def _create_melatonina_info(self):
        """Create information for melatonina when not found in CIMA"""
        return """
[Ref: Melatonina (Suplemento Dietético) (Nº Registro: custom_melatonina)]

INFORMACIÓN BÁSICA:
- Nombre: Melatonina
- Tipo: Suplemento alimenticio (no medicamento registrado en CIMA)
- Principios activos: Melatonina

DESCRIPCIÓN:
La melatonina es una hormona producida naturalmente por la glándula pineal en el cerebro. Como suplemento, se utiliza principalmente para trastornos del sueño y jet lag.

INDICACIONES HABITUALES:
- Trastornos del sueño
- Insomnio de corta duración
- Jet lag
- Trastornos del ritmo circadiano

POSOLOGÍA HABITUAL:
- Adultos: 0.5-5 mg antes de acostarse
- La dosis más común es 1-3 mg
- Se recomienda comenzar con la dosis más baja e ir aumentando según necesidad

PRECAUCIONES:
- No se recomienda su uso prolongado sin supervisión médica
- Puede producir somnolencia diurna
- No se recomienda en embarazo y lactancia
- Precaución en personas con enfermedades autoinmunes
- Puede interactuar con otros medicamentos

NOTA IMPORTANTE:
La melatonina se comercializa en España como complemento alimenticio, no como medicamento registrado en CIMA. La información proporcionada es orientativa y no sustituye la consulta con un profesional sanitario.
"""

    def _add_unique_results(self, new_results, all_results, seen_nregistros):
        """Add unique results to all_results list and return True if any were added"""
        added = False
        for med in new_results:
            if isinstance(med, dict) and med.get("nregistro"):
                nregistro = med.get("nregistro")
                if nregistro not in seen_nregistros:
                    seen_nregistros.add(nregistro)
                    all_results.append(med)
                    added = True
        return added

    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract potential search terms from the query"""
        # Common words to filter out
        common_words = {"sobre", "para", "como", "este", "esta", "estos", "estas", "cual", "cuales", 
                       "con", "por", "los", "las", "del", "que", "realizar", "realizar", "redactar", 
                       "crear", "generar", "prospecto", "formular", "elaborar", "realiza", "prepara"}
        
        # Extract individual words that might be medication names
        words = query.lower().split()
        potential_terms = []
        
        # Check our active principles database first
        for ap in self.active_principles:
            if ap in query.lower():
                potential_terms.append(ap)
        
        # Look for dosage patterns
        dosage_matches = re.findall(r'(\d+(?:[,.]\d+)?)\s*(%|mg|g|ml|mcg|UI|unidades)', query)
        for match in dosage_matches:
            potential_terms.append(match[0] + match[1])
        
        # Add remaining words if they seem significant
        for word in words:
            if len(word) > 4 and word not in common_words and word not in potential_terms:
                potential_terms.append(word)
        
        # Add important bigrams
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i+1]) > 3:
                bigram = f"{words[i]} {words[i+1]}"
                if bigram not in potential_terms:
                    potential_terms.append(bigram)
        
        # Remove duplicates
        return list(set(potential_terms))

    async def _search_in_specific_section(self, session, term, section):
        """Search for a term in a specific section of the technical data sheet"""
        search_url = f"{self.base_url}/buscarEnFichaTecnica"
        
        search_body = [{
            "seccion": section,
            "texto": term,
            "contiene": 1
        }]
        
        try:
            async with session.post(search_url, json=search_body) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, dict) and "resultados" in data:
                        return data.get("resultados", [])
        except Exception as e:
            logger.error(f"Error in section search for term {term} in section {section}: {str(e)}")
        
        return []

    async def _get_complete_medication_details(self, session, med: Dict, fetch_prospecto: bool = False, focus_section: str = None) -> str:
        """Get comprehensive details for a medication with enhanced section retrieval"""
        if not isinstance(med, dict):
            return ""
            
        # Handle custom medications (like melatonina)
        if med.get("custom_info") == True:
            if "melatonina" in med.get("nombre", "").lower():
                return self._create_melatonina_info()
            return f"Información personalizada para {med.get('nombre', 'medicamento')} no disponible en CIMA."
            
        nregistro = med.get("nregistro")
        if not nregistro:
            return ""
        
        # Defines API calls to make concurrently
        api_tasks = []
        
        # 1. Basic information
        async def get_basic_info():
            try:
                url = f"{self.base_url}/medicamento"
                async with session.get(url, params={"nregistro": nregistro}) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, dict):
                            return {"type": "basic", "data": result}
            except Exception as e:
                logger.error(f"Error getting basic info: {str(e)}")
            return {"type": "basic", "data": {}}
        
        api_tasks.append(get_basic_info())
        
        # 2. Define sections to retrieve, with focus section first if specified
        key_sections = {
            "4.1": "indicaciones",
            "4.2": "posologia",
            "4.3": "contraindicaciones",
            "4.4": "advertencias",
            "4.5": "interacciones",
            "4.6": "embarazo_lactancia",
            "4.8": "efectos_adversos",
            "5.1": "propiedades_farmacodinamicas",
            "5.2": "propiedades_farmacocineticas",
            "6.1": "excipientes"
        }
        
        # Reorder sections if focus section is specified
        if focus_section and focus_section in key_sections:
            focus_key = key_sections[focus_section]
            focused_sections = {focus_section: focus_key}
            for section, key in key_sections.items():
                if section != focus_section:
                    focused_sections[section] = key
            key_sections = focused_sections
        
        # Enhanced section retrieval with multiple fallback approaches
        async def get_section(section, key):
            # First approach - JSON API
            try:
                section_url = f"{self.base_url}/docSegmentado/contenido/1"
                params = {"nregistro": nregistro, "seccion": section}
                
                async with session.get(section_url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, dict) and "contenido" in result:
                            content = result.get("contenido", "")
                            if content and content != "No disponible":
                                return {"type": "section", "key": key, "data": content, "focused": section == focus_section}
            except Exception as e:
                logger.warning(f"Error in first approach for section {section}: {str(e)}")
            
            # Second approach - Direct HTML access
            try:
                section_html_url = f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/{section}/FichaTecnica.html"
                async with session.get(section_html_url) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        if html_content and len(html_content) > 50:  # Basic validity check
                            # Try to extract just text content
                            cleaned = re.sub(r'<[^>]+>', ' ', html_content)
                            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                            if cleaned:
                                return {"type": "section", "key": key, "data": cleaned, "focused": section == focus_section}
            except Exception as e:
                logger.warning(f"Error in second approach for section {section}: {str(e)}")
            
            # Third approach - Full technical sheet and extraction
            try:
                full_tech_url = f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html"
                async with session.get(full_tech_url) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        if html_content and len(html_content) > 200:
                            # Try to extract the specific section
                            section_pattern = f'<h3[^>]*>{section}\\.[^<]+</h3>(.*?)(?:<h3|<div class="section-break">)'
                            match = re.search(section_pattern, html_content, re.DOTALL)
                            if match:
                                section_content = match.group(1).strip()
                                # Clean HTML tags
                                cleaned = re.sub(r'<[^>]+>', ' ', section_content)
                                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                                if cleaned:
                                    return {"type": "section", "key": key, "data": cleaned, "focused": section == focus_section}
            except Exception as e:
                logger.warning(f"Error in third approach for section {section}: {str(e)}")
            
            # Fallback with unavailable message but include link
            return {"type": "section", "key": key, "data": f"No disponible. Consultar sección {section} en la ficha técnica completa: https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html", "focused": False}
        
        # Add section tasks
        section_tasks = [get_section(section, key) for section, key in key_sections.items()]
        api_tasks.extend(section_tasks)
        
        # 3. Get prospecto if requested
        if fetch_prospecto:
            async def get_prospecto():
                try:
                    # First approach - JSON API
                    url = f"{self.base_url}/docSegmentado/contenido/2"
                    params = {"nregistro": nregistro}
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            try:
                                result = await response.json()
                                if isinstance(result, dict) and "contenido" in result:
                                    content = result.get("contenido", "")
                                    if content and content != "No disponible":
                                        # Clean HTML tags for readability
                                        cleaned = re.sub(r'<[^>]+>', ' ', content)
                                        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                                        return {"type": "prospecto", "data": cleaned}
                            except:
                                # If JSON parse fails, try as text
                                content = await response.text()
                                if content and len(content) > 50:
                                    # Clean HTML tags
                                    cleaned = re.sub(r'<[^>]+>', ' ', content)
                                    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                                    return {"type": "prospecto", "data": cleaned}
                except Exception as e:
                    logger.warning(f"Error in prospecto retrieval: {str(e)}")
                
                # Second approach - Direct HTML access
                try:
                    prospecto_url = f"https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html"
                    async with session.get(prospecto_url) as response:
                        if response.status == 200:
                            html_content = await response.text()
                            if html_content and len(html_content) > 100:
                                # Extract just text content
                                cleaned = re.sub(r'<[^>]+>', ' ', html_content)
                                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                                return {"type": "prospecto", "data": cleaned}
                except Exception as e:
                    logger.warning(f"Error in direct prospecto HTML retrieval: {str(e)}")
                
                # Fallback with link
                return {"type": "prospecto", "data": f"Prospecto disponible en: https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html"}
            
            api_tasks.append(get_prospecto())
        
        # Execute all API calls concurrently
        api_results = await asyncio.gather(*api_tasks)
        
        # Process results
        basic_info = {}
        sections_data = {}
        focused_section_data = None
        prospecto_data = None
        
        for result in api_results:
            if result["type"] == "basic":
                basic_info = result["data"]
            elif result["type"] == "section":
                sections_data[result["key"]] = result["data"]
                # Save focused section separately
                if result.get("focused", False):
                    focused_section_data = (result["key"], result["data"])
            elif result["type"] == "prospecto":
                prospecto_data = result["data"]
        
        # Format into a readable text description
        return self._format_medication_details_text(
            med, basic_info, sections_data, nregistro, 
            prospecto_data, focused_section_data
        )
        
    def _format_medication_details_text(self, med, basic_info, sections_data, nregistro, prospecto_data=None, focused_section=None):
        """Format medication information with focus on the most relevant section"""
        # Basic details
        name = med.get("nombre", basic_info.get("nombre", "No disponible"))
        pactivos = med.get("pactivos", basic_info.get("pactivos", "No disponible"))
        lab = basic_info.get("labtitular", "No disponible")
        
        # Build the information block with most important sections first
        info_parts = [
            f"[Ref: {name} (Nº Registro: {nregistro})]",
            f"Nombre: {name}",
            f"Principios activos: {pactivos}",
            f"Laboratorio: {lab}"
        ]
        
        # If we have a focused section, add it first with emphasis
        if focused_section:
            key, content = focused_section
            section_title = {
                "indicaciones": "INDICACIONES TERAPÉUTICAS",
                "posologia": "POSOLOGÍA Y FORMA DE ADMINISTRACIÓN",
                "contraindicaciones": "CONTRAINDICACIONES",
                "advertencias": "ADVERTENCIAS Y PRECAUCIONES",
                "interacciones": "INTERACCIONES CON OTROS MEDICAMENTOS",
                "embarazo_lactancia": "EMBARAZO Y LACTANCIA",
                "efectos_adversos": "EFECTOS ADVERSOS",
                "propiedades_farmacodinamicas": "PROPIEDADES FARMACODINÁMICAS",
                "propiedades_farmacocineticas": "PROPIEDADES FARMACOCINÉTICAS",
                "excipientes": "EXCIPIENTES"
            }.get(key, key.upper())
            
            info_parts.append(f"### {section_title}:")
            info_parts.append(content)
        
        # Add other sections
        other_sections = {
            "indicaciones": "INDICACIONES TERAPÉUTICAS",
            "posologia": "POSOLOGÍA Y FORMA DE ADMINISTRACIÓN",
            "contraindicaciones": "CONTRAINDICACIONES",
            "advertencias": "ADVERTENCIAS Y PRECAUCIONES",
            "efectos_adversos": "EFECTOS ADVERSOS",
            "excipientes": "EXCIPIENTES"
        }
        
        for key, title in other_sections.items():
            # Skip if it's the focused section we already added
            if focused_section and key == focused_section[0]:
                continue
                
            content = sections_data.get(key, "")
            if content and len(content.strip()) > 0:
                info_parts.append(f"{title}:\n{content}")
        
        # Add prospecto if available
        if prospecto_data:
            info_parts.append("PROSPECTO (Información para el paciente):")
            info_parts.append(prospecto_data)
        
        # Add links to technical data sheet and prospecto
        info_parts.append(f"Ficha técnica completa: https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html")
        info_parts.append(f"Prospecto completo: https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html")
        
        return "\n\n".join(info_parts)

    async def chat(self, message: str) -> Dict[str, str]:
        """
        Handle chat messages with improved search and context management
        """
        try:
            # Analyze the query to determine key elements
            query_info = self._analyze_query(message)
            is_prospecto = query_info["is_prospecto"]
            
            # Get relevant information from CIMA with enhanced search
            context = await self.get_medication_info(message)
            
            # Choose the appropriate system prompt
            system_prompt = self.prospecto_prompt if is_prospecto else self.system_prompt
            
            # Count tokens for components
            system_tokens = self.num_tokens(system_prompt)
            message_tokens = self.num_tokens(message)
            context_tokens = self.num_tokens(context)
            
            logger.info(f"Token counts - System: {system_tokens}, Message: {message_tokens}, Context: {context_tokens}")
            
            # Process conversation history to stay within limits
            history_tokens = 0
            processed_history = []
            
            # Add as much history as will fit, starting from most recent
            for msg in reversed(self.conversation_history[-10:]):
                msg_tokens = self.num_tokens(msg["content"])
                if history_tokens + msg_tokens < 3000:  # Reserve token budget for history
                    processed_history.insert(0, msg)
                    history_tokens += msg_tokens
                else:
                    break
                    
            logger.info(f"History tokens: {history_tokens}, messages: {len(processed_history)}")
            
            # Enhanced prompt with query analysis for better responses
            prompt = f"""
Consulta: {message}

Análisis de la consulta:
- Principio activo detectado: {query_info['active_ingredient'] or 'No detectado'}
- Sección específica: {query_info['section_number'] or 'No especificada'}
- Tipo de petición: {"Prospecto" if is_prospecto else "Información general"}

Contexto relevante de CIMA:
{context if context else "No se encontró información específica en CIMA para esta consulta."}

{"Genera un prospecto completo siguiendo las directrices de la AEMPS." if is_prospecto else "Responde de manera detallada y precisa, citando las fuentes específicas del contexto."}

Si el principio activo es melatonina u otro que no esté registrado como medicamento en CIMA, explica brevemente que se trata de un suplemento alimenticio y proporciona información general sobre su uso habitual, destacando que no es un medicamento registrado.
"""
            
            prompt_tokens = self.num_tokens(prompt)
            total_tokens = system_tokens + history_tokens + prompt_tokens
            
            logger.info(f"Total input tokens: {total_tokens}")
            
            if total_tokens > 15000:
                logger.warning(f"Token count {total_tokens} approaching limit, might encounter issues")
            
            messages = [
                {"role": "system", "content": system_prompt},
                *processed_history,
                {"role": "user", "content": prompt}
            ]

            # Generate response 
            response = await self.openai_client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=messages,
                temperature=0.7
            )

            assistant_response = response.choices[0].message.content
            
            # Add direct links to CIMA
            pattern = r'\[Ref: ([^()]+) \(Nº Registro: (\d+)\)\]'
            
            def replace_with_link(match):
                med_name = match.group(1)
                reg_num = match.group(2)
                # Only create link if it's a valid registration number
                if reg_num and reg_num != "custom_melatonina" and not reg_num.startswith("custom_"):
                    return f'[Ref: {med_name} (Nº Registro: {reg_num})](https://cima.aemps.es/cima/dochtml/ft/{reg_num}/FT_{reg_num}.html)'
                return f'[Ref: {med_name}]'
            
            assistant_response_with_links = re.sub(pattern, replace_with_link, assistant_response)
            
            # Special handling for melatonina
            if "melatonina" in message.lower() and "custom_melatonina" in context:
                assistant_response_with_links = assistant_response_with_links.replace(
                    "[Ref: Melatonina (Suplemento Dietético) (Nº Registro: custom_melatonina)]", 
                    "**Nota sobre Melatonina:** La melatonina se comercializa en España como suplemento alimenticio, no como medicamento registrado en CIMA."
                )
            
            # Update conversation history
            self.conversation_history.extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": assistant_response}
            ])

            # Truncate context for UI display to improve performance
            display_context = context
            if len(display_context) > 10000:
                display_context = context[:9997] + "..."

            return {
                "answer": assistant_response_with_links,
                "context": display_context,
                "history": self.conversation_history
            }
        except Exception as e:
            error_message = f"Lo siento, ha ocurrido un error al procesar tu consulta: {str(e)}"
            logger.error(f"Error en chat: {str(e)}")
            return {
                "answer": error_message,
                "context": f"Error: {str(e)}",
                "history": self.conversation_history
            }

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        
    async def close(self):
        """Close the aiohttp session to free resources with proper error handling"""
        if self.session and not self.session.closed:
            try:
                await self.session.close()
                # Give the event loop time to clean up connections
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error closing session: {str(e)}")
                # Ensure session is marked as closed even if there was an error
                self.session = None