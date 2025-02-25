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
            try:
                async with session.get(tech_url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, dict):
                            return (key, result)
            except Exception as e:
                logger.error(f"Error retrieving section {section}: {str(e)}")
            return (key, {"contenido": f"No disponible"})
        
        async def get_prospecto_section(section=None):
            """Get prospecto content with improved error handling and multiple methods"""
            # First try XML format which is more reliable
            url = f"{self.base_url}/docSegmentado/contenido/2"
            params = {"nregistro": nregistro}
            if section:
                params["seccion"] = section
            
            # Try without specific Accept header first
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        try:
                            # Try to parse as JSON first
                            result = await response.json()
                            if isinstance(result, dict) and "contenido" in result:
                                return {"prospecto_html": result.get("contenido", "")}
                        except:
                            # If not JSON, try as text
                            content = await response.text()
                            return {"prospecto_html": content}
            except Exception as e:
                logger.error(f"Error in first prospecto retrieval attempt: {str(e)}")
            
            # Try with specific HTML Accept header
            try:
                headers = {"Accept": "text/html"}
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        return {"prospecto_html": content}
            except Exception as e:
                logger.error(f"Error in second prospecto retrieval attempt: {str(e)}")
            
            # Try alternative endpoint for prospecto
            try:
                alt_url = f"{self.base_url}/medicamento/{nregistro}/prospecto"
                async with session.get(alt_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        return {"prospecto_html": content}
            except Exception as e:
                logger.error(f"Error in alternative prospecto retrieval: {str(e)}")
            
            # Return a fallback to have at least something
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
        prospecto_pattern = r'(?:redactar|generar|crear|elaborar|realizar?e?|escrib[ei]r|hac[ae]r|desarroll[ae]r)\s+(?:un|el|uns?|una?)?\s+prospecto'
        
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
        
        # Extract active principles (assuming they're the first words or in uppercase)
        words = query.split()
        active_principle = words[0] if words else ""
        
        # Look for compound active principles (e.g., "Hidrocortisona y Lidocaína")
        compound_pattern = r'([A-Z][a-z]+(?:\s[a-z]+)*)\s+[y]\s+([A-Z][a-z]+(?:\s[a-z]+)*)'
        compound_match = re.search(compound_pattern, query)
        if compound_match:
            active_principle = f"{compound_match.group(1)} {compound_match.group(2)}"
        
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
        def get_section_content(section_key, max_len=400):
            section_data = details.get(section_key, {})
            if not isinstance(section_data, dict):
                return "No disponible"
            
            content = section_data.get('contenido', 'No disponible')
            # Truncate long content to save tokens
            if len(content) > max_len:
                return content[:max_len] + "..."
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
        Enhanced context retrieval with token limiting
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
        search_url = f"{self.base_url}/medicamentos"
        
        # Perform searches concurrently
        async def search_medications(params):
            try:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, dict) and "resultados" in data:
                            results = data.get("resultados", [])
                            logger.info(f"Search with params {params} returned {len(results)} results")
                            return results
            except Exception as e:
                logger.error(f"Error in search with params {params}: {str(e)}")
            return []
        
        # Execute the most important searches concurrently
        search_tasks = [
            search_medications({"nombre": query}),
            search_medications({"principiosActivos": active_principle}),
            search_medications({"formaFarmaceutica": formulation_type})
        ]
        
        search_results = await asyncio.gather(*search_tasks)
        
        # Combine results, avoiding duplicates
        all_results = []
        seen_nregistros = set()
        
        for results in search_results:
            for med in results:
                if isinstance(med, dict) and med.get("nregistro"):
                    nregistro = med.get("nregistro")
                    if nregistro not in seen_nregistros:
                        seen_nregistros.add(nregistro)
                        all_results.append(med)
        
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
        
        return "No se encontraron resultados relevantes."

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
                       "crear", "generar", "prospecto", "formular", "elaborar"}
        
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
        Optimized implementation for obtaining medication information with token limiting
        """
        cache_key = f"query_{query}"
        if cache_key in self.reference_cache:
            return self.reference_cache[cache_key]

        # Detect if this is a prospecto request
        prospecto_pattern = r'(?:redactar|generar|crear|elaborar|realizar?e?|escrib[ei]r|hac[ae]r|desarroll[ae]r)\s+(?:un|el|uns?|una?)?\s+prospecto'
        is_prospecto = bool(re.search(prospecto_pattern, query.lower()))
        logger.info(f"Query: '{query}', Is prospecto: {is_prospecto}")
        
        # Extract all potential search terms
        potential_terms = self._extract_search_terms(query)
        
        # Extract medication name from prospecto request
        if is_prospecto:
            # Try to extract medication name using patterns
            med_patterns = [
                r'(?:sobre|de|para|con)\s+(?:la|el|los|las)?\s*([a-zA-Z\-áéíóúÁÉÍÓÚñÑ]+(?:\s+[a-zA-Z\-áéíóúÁÉÍÓÚñÑ]+){0,3})\s+(\d+(?:[,.]\d+)?\s*(?:mg|g|ml|mcg|UI|%|unidades))',  # With concentration
                r'(?:sobre|de|para|con)\s+(?:la|el|los|las)?\s*([a-zA-Z\-áéíóúÁÉÍÓÚñÑ]+(?:\s+[a-zA-Z\-áéíóúÁÉÍÓÚñÑ]+){0,3})',  # Without concentration
                r'prospecto\s+(?:sobre|de|para|con)?\s+(?:la|el|los|las)?\s*([a-zA-Z\-áéíóúÁÉÍÓÚñÑ]+(?:\s+[a-zA-Z\-áéíóúÁÉÍÓÚñÑ]+){0,3})'  # After prospecto
            ]
            
            extracted_med_term = None
            for pattern in med_patterns:
                match = re.search(pattern, query.lower())
                if match:
                    extracted_term = match.group(1).strip()
                    if len(extracted_term) > 3:  # Avoid short/common words
                        extracted_med_term = extracted_term
                        logger.info(f"Extracted medication term from prospecto request: '{extracted_med_term}'")
                        break
                        
            # If we extracted a medication term, add it to the top of our search terms
            if extracted_med_term:
                potential_terms = [extracted_med_term] + [term for term in potential_terms if term != extracted_med_term]
        
        # Get the session
        session = await self.get_session()
        
        # Track already processed medications to avoid duplicates
        processed_nregistros = set()
        all_med_info = []
        
        # First try with full query if it's not a prospecto request
        if not is_prospecto:
            meds = await self._search_medications(session, query)
            for med in meds[:3]:  # Limit to 3 results for token management
                if med.get("nregistro") not in processed_nregistros:
                    processed_nregistros.add(med.get("nregistro"))
                    med_info = await self._get_complete_medication_details(session, med)
                    if med_info:
                        all_med_info.append(med_info)
        
        # For prospecto requests or if we didn't get results, focus on the extracted terms
        if is_prospecto or len(all_med_info) < 2:
            for term in potential_terms[:3]:  # Limit to first 3 terms
                term_meds = await self._search_medications(session, term)
                for med in term_meds[:2]:  # Limit to 2 results per term
                    if med.get("nregistro") not in processed_nregistros:
                        processed_nregistros.add(med.get("nregistro"))
                        med_info = await self._get_complete_medication_details(session, med, fetch_prospecto=is_prospecto)
                        if med_info:
                            all_med_info.append(med_info)
        
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
        
        # Store in cache
        self.reference_cache[cache_key] = combined_results
        return combined_results

    def _extract_search_terms(self, query: str) -> List[str]:
        """
        Extract all possible search terms from the query
        """
        # Patterns for different types of potential search terms
        patterns = [
            r'([A-Z][a-záéíóúñ]+(?:\s[a-záéíóúñ]+){0,3})',  # Capitalized words
            r'(\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|UI|unidades))',  # Dosages
            r'([A-Za-záéíóúñ]+\+[A-Za-záéíóúñ]+)'  # Combinations with +
        ]
        
        # List of common stopwords to filter out
        stopwords = {"sobre", "para", "como", "este", "esta", "estos", "estas", "cual", "cuales", 
                    "con", "por", "los", "las", "del", "que", "realizar", "realizar", "redactar", 
                    "crear", "generar", "prospecto", "formular", "elaborar"}
        
        # Extract using patterns
        potential_terms = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            potential_terms.extend([m.strip() for m in matches if len(m.strip()) > 3])
        
        # Extract individual words
        words = query.split()
        for word in words:
            if len(word) > 4 and word.lower() not in stopwords:
                potential_terms.append(word)
        
        # Add bi-grams (pairs of words)
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i+1]) > 3:
                bigram = f"{words[i]} {words[i+1]}"
                potential_terms.append(bigram)
        
        # Extract medication categories and conditions
        categories = [
            "analgésico", "antibiótico", "antiinflamatorio", "antidepresivo", 
            "ansiolítico", "antihistamínico", "antihipertensivo", "hipnótico",
            "antiácido", "anticoagulante", "antidiabético", "antipsicótico",
            "corticoide", "diurético", "laxante", "mucolítico"
        ]
        
        conditions = [
            "hipertensión", "diabetes", "asma", "ansiedad", "depresión",
            "dolor", "infección", "alergia", "insomnio", "artritis",
            "migraña", "úlcera", "reflujo", "colesterol", "epilepsia"
        ]
        
        # Add any matching categories and conditions
        query_lower = query.lower()
        for term in categories + conditions:
            if term in query_lower:
                potential_terms.append(term)
        
        # Remove duplicates
        seen = set()
        return [x for x in potential_terms if x.lower() not in seen and not seen.add(x.lower())]

    async def _search_medications(self, session, query: str) -> List[Dict]:
        """
        Search medications with multiple strategies
        """
        search_url = f"{self.base_url}/medicamentos"
        all_results = []
        
        # Search strategies in order of relevance
        search_strategies = [
            {"params": {"nombre": query}},
            {"params": {"practiv1": query}},
            {"params": {"atc": query}}
        ]
        
        # Execute searches concurrently
        async def execute_search(params):
            try:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, dict) and "resultados" in data:
                            return data.get("resultados", [])
            except Exception as e:
                logger.error(f"Error in medication search: {str(e)}")
            return []
        
        # Run all searches concurrently
        search_tasks = [execute_search(strategy["params"]) for strategy in search_strategies]
        search_results = await asyncio.gather(*search_tasks)
        
        # Combine results, avoiding duplicates
        seen_nregistros = set()
        
        for results in search_results:
            for med in results:
                if isinstance(med, dict) and med.get("nregistro"):
                    nregistro = med.get("nregistro")
                    if nregistro not in seen_nregistros:
                        seen_nregistros.add(nregistro)
                        all_results.append(med)
        
        # If no results and query is longer than 4 chars, try partial match
        if not all_results and len(query) > 4:
            partial_query = query[:4]
            partial_results = await execute_search({"nombre": partial_query})
            
            for med in partial_results:
                if isinstance(med, dict) and med.get("nregistro"):
                    nregistro = med.get("nregistro")
                    if nregistro not in seen_nregistros:
                        seen_nregistros.add(nregistro)
                        all_results.append(med)
        
        return all_results

    async def _search_in_ficha_tecnica(self, session, query: str) -> List[Dict]:
        """
        Search in technical files for more comprehensive results
        """
        results = []
        search_url = f"{self.base_url}/buscarEnFichaTecnica"
        
        # Focus on most important sections for token efficiency
        sections = ["4.1", "4.2", "4.3"]
        
        # Extract key words for search
        words = [word for word in query.split() if len(word) > 3]
        if not words:
            return []
            
        # Limit to 2 words for performance and token management
        words = words[:2]
        
        search_body = []
        for section in sections:
            for word in words:
                search_body.append({
                    "seccion": section,
                    "texto": word,
                    "contiene": 1
                })
        
        try:
            async with session.post(search_url, json=search_body) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, dict) and "resultados" in data:
                        results = data.get("resultados", [])
        except Exception as e:
            logger.error(f"Error in ficha técnica search: {str(e)}")
        
        return results

    async def _get_complete_medication_details(self, session, med: Dict, fetch_prospecto: bool = False) -> str:
        """
        Get comprehensive details for a medication with token efficiency
        """
        if not isinstance(med, dict) or not med.get("nregistro"):
            return ""
        
        nregistro = med.get("nregistro")
        
        # Track all API calls to make concurrently
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
        
        # 2. Most important technical sections only
        key_sections = {
            "4.1": "indicaciones",
            "4.2": "posologia",
            "4.3": "contraindicaciones",
            "4.4": "advertencias",
            "4.8": "efectos_adversos",
            "6.1": "excipientes"
        }
        
        async def get_section(section, key):
            try:
                section_url = f"{self.base_url}/docSegmentado/contenido/1"
                async with session.get(section_url, params={"nregistro": nregistro, "seccion": section}) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, dict) and "contenido" in result:
                            content = result.get("contenido", "")
                            # Truncate long content to save tokens
                            if len(content) > 400:
                                content = content[:397] + "..."
                            return {"type": "section", "key": key, "data": content}
            except Exception as e:
                logger.error(f"Error getting section {section}: {str(e)}")
            return {"type": "section", "key": key, "data": "No disponible"}
        
        for section, key in key_sections.items():
            api_tasks.append(get_section(section, key))
        
        # 3. Prospecto if requested
        if fetch_prospecto:
            async def get_prospecto():
                try:
                    # First try XML format which is more reliable
                    url = f"{self.base_url}/docSegmentado/contenido/2"
                    params = {"nregistro": nregistro}
                    
                    # Try without specific Accept header first
                    headers = {}
                    async with session.get(url, params=params, headers=headers) as response:
                        if response.status == 200:
                            try:
                                # Try to parse as JSON first
                                result = await response.json()
                                if isinstance(result, dict) and "contenido" in result:
                                    content = result.get("contenido", "")
                                    # Clean and truncate
                                    cleaned = re.sub(r'<[^>]+>', ' ', content)
                                    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                                    if len(cleaned) > 800:
                                        cleaned = cleaned[:797] + "..."
                                    return {"type": "prospecto", "data": cleaned}
                            except:
                                # If not JSON, try as text
                                content = await response.text()
                                cleaned = re.sub(r'<[^>]+>', ' ', content)
                                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                                if len(cleaned) > 800:
                                    cleaned = cleaned[:797] + "..."
                                return {"type": "prospecto", "data": cleaned}
                                
                    # Try alternative endpoint for prospecto
                    alt_url = f"{self.base_url}/medicamento/{nregistro}/prospecto"
                    async with session.get(alt_url) as response:
                        if response.status == 200:
                            try:
                                result = await response.json()
                                if isinstance(result, dict) and "prospecto" in result:
                                    content = result.get("prospecto", "")
                                    cleaned = re.sub(r'<[^>]+>', ' ', content)
                                    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                                    if len(cleaned) > 800:
                                        cleaned = cleaned[:797] + "..."
                                    return {"type": "prospecto", "data": cleaned}
                            except:
                                pass
                except Exception as e:
                    logger.error(f"Error getting prospecto for {nregistro}: {str(e)}")
                
                # Fallback to a simple URL reference if content retrieval fails
                return {"type": "prospecto", "data": f"Prospecto disponible en: https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html"}
                
            api_tasks.append(get_prospecto())
        
        # Execute all API calls concurrently
        api_results = await asyncio.gather(*api_tasks)
        
        # Process results
        basic_info = {}
        sections_data = {}
        prospecto_data = None
        
        for result in api_results:
            if result["type"] == "basic":
                basic_info = result["data"]
            elif result["type"] == "section":
                sections_data[result["key"]] = result["data"]
            elif result["type"] == "prospecto":
                prospecto_data = result["data"]
        
        # Format into a token-efficient text description
        return self._format_medication_details_text(med, basic_info, sections_data, nregistro, prospecto_data)
        
    def _format_medication_details_text(self, med, basic_info, sections_data, nregistro, prospecto_data=None):
        """Format medication details with token efficiency"""
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
        
        # Add key sections, most important first
        for key, title in {
            "indicaciones": "INDICACIONES TERAPÉUTICAS",
            "posologia": "POSOLOGÍA Y FORMA DE ADMINISTRACIÓN",
            "contraindicaciones": "CONTRAINDICACIONES",
            "advertencias": "ADVERTENCIAS Y PRECAUCIONES",
            "efectos_adversos": "EFECTOS ADVERSOS",
            "excipientes": "EXCIPIENTES"
        }.items():
            content = sections_data.get(key, "")
            if content and len(content.strip()) > 0:
                info_parts.append(f"{title}:\n{content}")
        
        # Add prospecto if available (only for prospecto requests)
        if prospecto_data and prospecto_data != "No disponible":
            info_parts.append("PROSPECTO (Información para el paciente):")
            info_parts.append(prospecto_data)
        
        # Add links to technical data sheet and prospecto
        info_parts.append(f"Ficha técnica completa: https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html")
        info_parts.append(f"Prospecto completo: https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html")
        
        return "\n\n".join(info_parts)

    def _format_date(self, unix_timestamp):
        """Format dates from Unix timestamp to readable format"""
        if not unix_timestamp:
            return "No disponible"
        
        try:
            # Dates in the CIMA API are in Unix timestamp (milliseconds)
            dt = datetime.fromtimestamp(unix_timestamp / 1000)
            return dt.strftime("%d/%m/%Y")
        except:
            return str(unix_timestamp)

    async def chat(self, message: str) -> Dict[str, str]:
        """
        Handle chat messages with token management
        """
        try:
            # Check if this is a prospecto request
            prospecto_pattern = r'(?:redactar|generar|crear|elaborar|realizar?e?|escrib[ei]r|hac[ae]r|desarroll[ae]r)\s+(?:un|el|uns?|una?)?\s+prospecto'
            is_prospecto = bool(re.search(prospecto_pattern, message.lower()))
            
            # Get relevant information from CIMA
            context = await self.get_medication_info(message)
            
            # If no results, try with generic terms
            if not context:
                generic_terms = self._extract_search_terms(message)
                for term in generic_terms:
                    term_context = await self.get_medication_info(term)
                    if term_context:
                        context = term_context
                        break
            
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
            
            # Prepare the prompt
            prompt = f"""
Consulta: {message}

Contexto relevante de CIMA:
{context if context else "No se encontró información específica en CIMA para esta consulta."}

{"Genera un prospecto completo siguiendo las directrices de la AEMPS." if is_prospecto else "Responde de manera detallada y precisa, citando las fuentes específicas del contexto."}
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
                return f'[Ref: {med_name} (Nº Registro: {reg_num})](https://cima.aemps.es/cima/dochtml/ft/{reg_num}/FT_{reg_num}.html)'
            
            assistant_response_with_links = re.sub(pattern, replace_with_link, assistant_response)
            
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