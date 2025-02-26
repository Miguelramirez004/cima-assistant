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

Basa toda la información en los datos proporcionados en el contexto CIMA, citando apropiadamente las fuentes con el formato [Ref: Nombre del medicamento (Nº Registro)].

Si no hay información disponible en CIMA sobre el medicamento solicitado, proporciona una explicación clara y ofrece sugerencias alternativas basadas en conocimientos generales. Indica claramente que esta información no proviene de CIMA y que debe ser verificada por un profesional sanitario.
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
        prospecto_pattern = r'(?:redactar|generar|crear|elaborar|realizar?e?|escrib[ei]r|hac[ae]r|desarroll[ae]r|realizar?|prosp)\s+(?:un|el|uns?|una?)?\s+(?:prospecto|prosp)'
        
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
            
            # If no active ingredient found, extract from the query directly
            if not active_principle or active_principle == "":
                words = query.split()
                for word in words:
                    if len(word) > 4 and word.lower() not in {"prospecto", "generar", "crear", "realizar", "redactar", "elaborar"}:
                        active_principle = word
                        logger.info(f"Extracted potential medication term from query: '{active_principle}'")
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
        
        # Build search terms from the query
        search_terms = self._extract_search_terms(query)
        if active_principle and active_principle not in search_terms and len(active_principle) > 3:
            search_terms.insert(0, active_principle)  # Add active principle to the beginning if not already included
        
        logger.info(f"Search terms extracted: {search_terms}")
        
        # Get common medication search terms
        common_medications = self._get_common_medications()
        found_medications = [term for term in search_terms if term.lower() in common_medications]
        
        # Perform searches concurrently
        async def search_medications(params):
            try:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, dict) and "resultados" in data:
                            results = data.get("resultados", [])
                            logger.info(f"Search with params {params} returned {len(results)} results")
                            # Filter results to ensure they are relevant
                            if params.get("principiosActivos") or params.get("nombre"):
                                search_term = params.get("principiosActivos", params.get("nombre", "")).lower()
                                filtered_results = self._filter_relevant_results(results, search_term)
                                return filtered_results
                            return results
            except Exception as e:
                logger.error(f"Error in search with params {params}: {str(e)}")
            return []
        
        # Build search tasks prioritizing more specific searches
        search_tasks = []
        
        # Primary search: by active principle or medication name
        if found_medications:
            for med in found_medications[:2]:  # Limit to top 2 known medications
                search_tasks.append(search_medications({"principiosActivos": med}))
                search_tasks.append(search_medications({"nombre": med}))
        
        # Secondary search: by all extracted terms
        for term in search_terms[:3]:  # Limit to top 3 terms
            search_tasks.append(search_medications({"principiosActivos": term}))
            search_tasks.append(search_medications({"nombre": term}))
        
        # Tertiary search: by formulation type if we have an identifiable one
        if formulation_type:
            search_tasks.append(search_medications({"formaFarmaceutica": formulation_type}))
        
        # Execute searches concurrently
        search_results = await asyncio.gather(*search_tasks)
        
        # Combine results, avoiding duplicates and prioritizing by relevance
        all_results = []
        seen_nregistros = set()
        
        # Function to evaluate relevance to the query
        def relevance_score(med, query_terms):
            score = 0
            
            # Get the medication name and active ingredients
            med_name = med.get('nombre', '').lower()
            pactivos = med.get('pactivos', '').lower()
            
            # Check each term for presence in the medication name or active ingredients
            for term in query_terms:
                term = term.lower()
                if term in med_name:
                    score += 10  # Higher score for match in name
                if term in pactivos:
                    score += 8   # Good score for match in active ingredients
                
            # Prioritize medications that contain the exact term
            for term in query_terms:
                if term.lower() == med_name.lower() or term.lower() == pactivos.lower():
                    score += 20  # Exact match gets a big boost
                    
            return score
        
        # Process search results and sort by relevance
        scored_results = []
        for results in search_results:
            for med in results:
                if isinstance(med, dict) and med.get("nregistro"):
                    nregistro = med.get("nregistro")
                    if nregistro not in seen_nregistros:
                        seen_nregistros.add(nregistro)
                        score = relevance_score(med, search_terms)
                        scored_results.append((med, score))
        
        # Sort by relevance score (highest first)
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top results
        all_results = [med for med, score in scored_results if score > 0][:n_results]
        
        # If we still don't have enough results, add from the original results
        if len(all_results) < n_results:
            for results in search_results:
                for med in results:
                    if isinstance(med, dict) and med.get("nregistro"):
                        nregistro = med.get("nregistro")
                        if nregistro not in seen_nregistros:
                            seen_nregistros.add(nregistro)
                            all_results.append(med)
                            if len(all_results) >= n_results:
                                break
                if len(all_results) >= n_results:
                    break
        
        # Limit results to requested number
        results = all_results[:n_results]
        cached_results = []

        if results:
            # First verify if any of the results are actually relevant
            relevant_results = [med for med in results if self._is_relevant_medication(med, query, search_terms)]
            
            if not relevant_results:
                # No relevant results found
                if is_prospecto:
                    return self._generate_not_found_message(active_principle, formulation_info["concentration"])
                return "No se encontraron medicamentos relevantes para esta consulta en la base de datos CIMA."
            
            logger.info(f"Found {len(relevant_results)} relevant medications, fetching details")
            
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
            detail_tasks = [limited_fetch_med_details(med) for med in relevant_results]
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
        
        if is_prospecto:
            return self._generate_not_found_message(active_principle, formulation_info["concentration"])
        
        return "No se encontraron medicamentos relevantes para esta consulta en la base de datos CIMA."

    def _filter_relevant_results(self, results, search_term):
        """Filter to keep only relevant results based on the search term"""
        relevant_results = []
        for med in results:
            med_name = med.get('nombre', '').lower()
            pactivos = med.get('pactivos', '').lower()
            
            # Only include if the search term appears in the name or active ingredients
            if search_term in med_name or search_term in pactivos:
                relevant_results.append(med)
        
        return relevant_results if relevant_results else results[:3]  # Return at least some results

    def _is_relevant_medication(self, med, query, search_terms):
        """Determine if a medication is actually relevant to the query"""
        query_lower = query.lower()
        med_name = med.get('nombre', '').lower()
        pactivos = med.get('pactivos', '').lower()
        
        # Blacklist certain terms that often appear as default/first results
        blacklist = ["abacavir", "vitamina", "abacavir/lamivudina", "acenocumarol"]
        
        # Check if this is a blacklisted medication with no explicit match
        for term in blacklist:
            if term in med_name.lower() or term in pactivos.lower():
                # Check if any search term explicitly matches this medication
                if not any(term.lower() in med_name.lower() or term.lower() in pactivos.lower() for term in search_terms):
                    return False
        
        # Check if any search term appears in the medication name or active ingredients
        for term in search_terms:
            term_lower = term.lower()
            if term_lower in med_name or term_lower in pactivos:
                return True
                
        # Check if any common medication term appears in both query and medication
        common_meds = self._get_common_medications()
        for med_term in common_meds:
            if med_term in query_lower and (med_term in med_name or med_term in pactivos):
                return True
                
        # For cases where no explicit match, check if there's some partial overlap
        med_words = set(med_name.split() + pactivos.split())
        query_words = set(query_lower.split())
        common_words = med_words.intersection(query_words)
        if len(common_words) > 0 and not all(word in ["de", "la", "el", "los", "las", "para", "con", "sin"] for word in common_words):
            return True
            
        return False

    def _generate_not_found_message(self, active_principle, concentration):
        """Generate a helpful message when no results are found for a prospecto request"""
        message = f"No se encontró información específica sobre "
        
        if active_principle and active_principle != "":
            message += f"{active_principle}"
            if concentration:
                message += f" {concentration}"
            message += " "
        
        message += "en la base de datos CIMA. "
        
        message += """
Es posible que este medicamento no esté registrado en la AEMPS, que se comercialice bajo otro nombre, o que sea un complemento alimenticio/producto sanitario en lugar de un medicamento.

Para obtener información precisa:
1. Verifique la ortografía del nombre del medicamento
2. Intente buscar por el nombre del principio activo
3. Consulte el prospecto físico del medicamento
4. Consulte con su farmacéutico o médico

Si necesita un prospecto general para este principio activo, indique el nombre comercial específico o el laboratorio fabricante para obtener resultados más precisos.
"""
        return message

    def _get_common_medications(self):
        """Return a list of common medications in Spanish for better matching"""
        return [
            "melatonina", "ibuprofeno", "paracetamol", "omeprazol", "amoxicilina", 
            "simvastatina", "enalapril", "metformina", "lorazepam", "diazepam", 
            "fluoxetina", "atorvastatina", "tramadol", "naproxeno", "metamizol", 
            "aspirina", "ácido acetilsalicílico", "acetilsalicilico", "salbutamol", 
            "fluticasona", "amlodipino", "valsartan", "losartan", "dexametasona", 
            "betametasona", "fentanilo", "morfina", "alendronato", "quetiapina", 
            "risperidona", "levotiroxina", "ranitidina", "cetirizina", "ebastina",
            "ambroxol", "ciprofloxacino", "levofloxacino", "azitromicina",
            "amlodipino", "hidroclorotiazida", "venlafaxina", "sertralina",
            "escitalopram", "pantoprazol", "lansoprazol", "furosemida",
            "alprazolam", "metoclopramida", "hidrocortisona", "aciclovir",
            "valaciclovir", "claritromicina", "ciclobenzaprina", "prednisona",
            "prednisolona", "budesonida", "montelukast", "formoterol"
        ]

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
        
        # Check for common medications in the query
        common_meds = self._get_common_medications()
        for med in common_meds:
            if med in query.lower() and med not in [term.lower() for term in potential_terms]:
                potential_terms.append(med)
        
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

Si no hay información específica sobre el medicamento solicitado en el contexto CIMA, explica claramente que no se encuentra disponible en la base de datos y proporciona recomendaciones generales basadas en conocimientos farmacéuticos.
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

Si el medicamento solicitado no se encuentra en la base de datos CIMA, explica esto claramente y proporciona información general sobre el tipo de medicamento cuando sea posible, indicando claramente que esta información no proviene de CIMA.
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

Si no hay información disponible en CIMA sobre el medicamento solicitado, explica claramente que no se encuentra en la base de datos y proporciona información general sobre el principio activo cuando sea posible, indicando claramente que esta información no proviene de CIMA y que debe ser verificada por un profesional sanitario.
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
        
        # Track already processed medications to avoid duplicates
        processed_nregistros = set()
        all_med_info = []

        # Get common medication names for better matching
        common_medications = self._get_common_medications()
        requested_med = query_info["active_ingredient"]
        
        # 1. Try direct searches using the active ingredient and section-specific searches
        if query_info["active_ingredient"]:
            # If we have a section to focus on, use the buscarEnFichaTecnica endpoint
            if query_info["section_number"]:
                search_results = await self._search_in_specific_section(
                    session, 
                    query_info["active_ingredient"], 
                    query_info["section_number"]
                )
                
                # Verify relevance of the results
                search_results = [med for med in search_results if self._is_relevant_medication(
                    med, 
                    query, 
                    [query_info["active_ingredient"]] + query_info["search_terms"]
                )]
                
                # Process the results
                for med in search_results:
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
            
            # Standard search by active ingredient
            if len(all_med_info) < 2:
                # First try with the most specific search
                meds = await self._search_medications(session, {"principiosActivos": query_info["active_ingredient"]})
                
                # Filter out irrelevant results
                meds = [med for med in meds if self._is_relevant_medication(
                    med, 
                    query, 
                    [query_info["active_ingredient"]] + query_info["search_terms"]
                )]
                
                for med in meds[:3]:  # Limit to 3 results
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
        
        # 2. If we still don't have enough results, try with the general search terms
        if len(all_med_info) < 2:
            for term in query_info["search_terms"][:3]:  # Limit to first 3 terms
                if term.lower() in common_medications:  # Only search for terms that might be medications
                    term_meds = await self._search_medications(session, {"nombre": term})
                    term_meds = [med for med in term_meds if self._is_relevant_medication(med, query, [term] + query_info["search_terms"])]
                    
                    for med in term_meds[:2]:  # Limit to 2 results per term
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
                                
                # Break if we have enough results
                if len(all_med_info) >= 2:
                    break
        
        # 3. If we still don't have results and have a specific medication, provide a not found message
        if len(all_med_info) == 0 and requested_med:
            not_found_message = self._generate_not_found_message(
                requested_med, 
                query_info["is_prospecto"], 
                query_info["concentration"]
            )
            
            # Store in cache
            self.reference_cache[cache_key] = not_found_message
            return not_found_message
        
        # 4. If we still don't have results, provide generic information
        if len(all_med_info) == 0:
            generic_message = (
                "No se encontraron medicamentos relevantes para esta consulta en la base de datos CIMA. "
                "Para obtener información específica, intente incluir el nombre exacto del medicamento o el principio activo "
                "en su consulta. También puede especificar la concentración o la forma farmacéutica para mejorar los resultados."
            )
            
            # Store in cache
            self.reference_cache[cache_key] = generic_message
            return generic_message
        
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

    def _generate_not_found_message(self, medication_name, is_prospecto=False, concentration=None):
        """Generate a helpful message when a specific medication is not found"""
        concentration_str = f" {concentration}" if concentration else ""
        
        if is_prospecto:
            return f"""
# Información sobre "{medication_name}{concentration_str}"

Lo sentimos, no se encontró información sobre "{medication_name}{concentration_str}" en la base de datos CIMA de la AEMPS.

## Posibles razones:
1. Este producto podría ser un complemento alimenticio o producto sanitario, no un medicamento registrado
2. Podría estar registrado con otro nombre comercial
3. Podría no estar comercializado en España
4. Podría haber un error en el nombre o en la concentración

## Recomendaciones:
- Verifique la ortografía exacta del medicamento
- Intente buscar por el principio activo en lugar del nombre comercial
- Consulte con su farmacéutico para obtener información específica
- Verifique si se trata de un complemento alimenticio (no registrado en CIMA)

Para obtener un prospecto válido, necesitamos el nombre exacto de un medicamento registrado en la AEMPS.
"""
        else:
            return f"""
No se encontró información sobre "{medication_name}{concentration_str}" en la base de datos CIMA de la AEMPS.

Posibles razones por las que no aparece este medicamento:
1. Podría ser un complemento alimenticio o producto sanitario (no registrado como medicamento)
2. Podría estar registrado con otro nombre comercial
3. Podría no estar comercializado en España
4. Podría haber un error en el nombre o en la concentración

Para obtener información específica, intente:
- Verificar la ortografía exacta del medicamento
- Buscar por el principio activo en lugar del nombre comercial
- Consultar con su farmacéutico
- Especificar el laboratorio fabricante
"""

    def _get_common_medications(self):
        """Return a list of common medications in Spanish for better matching"""
        return [
            "melatonina", "ibuprofeno", "paracetamol", "omeprazol", "amoxicilina", 
            "simvastatina", "enalapril", "metformina", "lorazepam", "diazepam", 
            "fluoxetina", "atorvastatina", "tramadol", "naproxeno", "metamizol", 
            "aspirina", "ácido acetilsalicílico", "acetilsalicilico", "salbutamol", 
            "fluticasona", "amlodipino", "valsartan", "losartan", "dexametasona", 
            "betametasona", "fentanilo", "morfina", "alendronato", "quetiapina", 
            "risperidona", "levotiroxina", "ranitidina", "cetirizina", "ebastina",
            "ambroxol", "ciprofloxacino", "levofloxacino", "azitromicina",
            "amlodipino", "hidroclorotiazida", "venlafaxina", "sertralina",
            "escitalopram", "pantoprazol", "lansoprazol", "furosemida",
            "alprazolam", "metoclopramida", "hidrocortisona", "aciclovir",
            "valaciclovir", "claritromicina", "ciclobenzaprina", "prednisona",
            "prednisolona", "budesonida", "montelukast", "formoterol"
        ]

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Enhanced query analysis to extract key information
        """
        query_lower = query.lower()
        is_prospecto = bool(re.search(r'(?:prospecto|reda[ckt]|crear?|generar?|realizar?)\s+(?:un|el|de|sobre)', query_lower))
        
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
                
        # Extract active ingredient from common medications
        common_medications = self._get_common_medications()
        
        # Look for concentration patterns
        concentration_pattern = r'(\d+(?:[,.]\d+)?)\s*(%|mg|g|ml|mcg|UI|unidades)'
        concentration_match = re.search(concentration_pattern, query)
        concentration = concentration_match.group(0) if concentration_match else None
        
        # Try to find a medication name in the query
        active_ingredient = None
        for med in common_medications:
            if med in query_lower:
                active_ingredient = med
                break
        
        # Extract all potential search terms
        search_terms = self._extract_search_terms(query)
        
        # If active ingredient wasn't found through common medications, try to extract it from search terms
        if not active_ingredient and search_terms:
            # Check if any search term is a known medication
            med_terms = [term for term in search_terms if term.lower() in common_medications]
            if med_terms:
                active_ingredient = med_terms[0]
            else:
                # Use the first search term as a potential medication name
                active_ingredient = search_terms[0]
        
        return {
            "active_ingredient": active_ingredient,
            "section_number": section_number,
            "is_prospecto": is_prospecto,
            "search_terms": search_terms,
            "concentration": concentration
        }

    async def _search_in_specific_section(self, session, term, section):
        """
        Search in a specific section of the technical data sheet using the buscarEnFichaTecnica endpoint
        """
        search_url = f"{self.base_url}/buscarEnFichaTecnica"
        
        # Create the search body JSON
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
        
        # List of medical/pharmaceutical section terms to filter out
        section_terms = {"indicacion", "indicaciones", "posologia", "dosis", "contraindicacion", 
                         "contraindicaciones", "advertencia", "advertencias", "precaucion", 
                         "precauciones", "interaccion", "interacciones", "efecto", "efectos", 
                         "adverso", "adversos", "secundario", "secundarios"}
        
        # Extract using patterns
        potential_terms = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            potential_terms.extend([m.strip() for m in matches if len(m.strip()) > 3])
        
        # Extract individual words
        words = query.split()
        for word in words:
            if (len(word) > 4 and 
                word.lower() not in stopwords and 
                word.lower() not in section_terms and
                word not in potential_terms):
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

    async def _search_medications(self, session, params: Dict = None) -> List[Dict]:
        """
        Enhanced medication search with multiple fallback strategies
        """
        if params is None:
            params = {}
            
        search_url = f"{self.base_url}/medicamentos"
        all_results = []
        
        # Define blacklisted medications that often appear as first results
        blacklist = ["abacavir", "lamivudina", "abacavir/lamivudina", "acenocumarol"]
        
        # Function to filter out blacklisted medications when not explicitly requested
        def filter_blacklisted(results, search_term=None):
            if not search_term:
                return [med for med in results if not any(
                    black in med.get('nombre', '').lower() or black in med.get('pactivos', '').lower() 
                    for black in blacklist)]
            else:
                # Don't filter if the search term matches a blacklisted item
                search_term = search_term.lower()
                if any(black in search_term for black in blacklist):
                    return results
                # Otherwise, filter out blacklisted items
                return [med for med in results if not any(
                    black in med.get('nombre', '').lower() or black in med.get('pactivos', '').lower() 
                    for black in blacklist)]
        
        # Execute the primary search
        search_term = params.get("principiosActivos", params.get("nombre", ""))
        try:
            async with session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, dict) and "resultados" in data:
                        results = data.get("resultados", [])
                        
                        # Apply blacklist filtering
                        filtered_results = filter_blacklisted(results, search_term)
                        
                        logger.info(f"Search with params {params} returned {len(results)} results, {len(filtered_results)} after filtering")
                        all_results.extend(filtered_results)
        except Exception as e:
            logger.error(f"Error in medication search with params {params}: {str(e)}")
        
        # If no results and searching by name with more than 4 characters, try partial match
        if not all_results and params.get("nombre") and len(params["nombre"]) > 4:
            partial_query = params["nombre"][:4]
            try:
                async with session.get(search_url, params={"nombre": partial_query}) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, dict) and "resultados" in data:
                            results = data.get("resultados", [])
                            # Apply blacklist filtering
                            filtered_results = filter_blacklisted(results, partial_query)
                            logger.info(f"Partial search with term {partial_query} returned {len(results)} results, {len(filtered_results)} after filtering")
                            all_results.extend(filtered_results)
            except Exception as e:
                logger.error(f"Error in partial match search: {str(e)}")
        
        # If still no results and searching by principle active, try a more general search
        if not all_results and params.get("principiosActivos"):
            try:
                # Try to search just by the first word of the principle active
                first_word = params["principiosActivos"].split()[0]
                if len(first_word) > 3:
                    async with session.get(search_url, params={"practiv1": first_word}) as response:
                        if response.status == 200:
                            data = await response.json()
                            if isinstance(data, dict) and "resultados" in data:
                                results = data.get("resultados", [])
                                # Apply blacklist filtering
                                filtered_results = filter_blacklisted(results, first_word)
                                logger.info(f"Fallback search with practiv1={first_word} returned {len(results)} results, {len(filtered_results)} after filtering")
                                all_results.extend(filtered_results)
            except Exception as e:
                logger.error(f"Error in fallback principle active search: {str(e)}")
        
        # Avoid duplicates by keeping track of unique nregistro values
        seen_nregistros = set()
        unique_results = []
        
        for med in all_results:
            if isinstance(med, dict) and med.get("nregistro"):
                nregistro = med.get("nregistro")
                if nregistro not in seen_nregistros:
                    seen_nregistros.add(nregistro)
                    unique_results.append(med)
                    
        return unique_results

    def _is_relevant_medication(self, med, query, search_terms):
        """Determine if a medication is actually relevant to the query"""
        query_lower = query.lower()
        med_name = med.get('nombre', '').lower()
        pactivos = med.get('pactivos', '').lower()
        
        # Blacklist certain terms that often appear as default/first results
        blacklist = ["abacavir", "lamivudina", "abacavir/lamivudina", "acenocumarol"]
        
        # Check if this is a blacklisted medication with no explicit match
        for term in blacklist:
            if term in med_name.lower() or term in pactivos.lower():
                # Check if any search term explicitly matches this medication
                if not any(term.lower() in med_name.lower() or term.lower() in pactivos.lower() for term in search_terms):
                    return False
        
        # Check if any search term appears in the medication name or active ingredients
        for term in search_terms:
            term_lower = term.lower()
            if term_lower in med_name or term_lower in pactivos:
                return True
                
        # Check if any common medication term appears in both query and medication
        common_meds = self._get_common_medications()
        for med_term in common_meds:
            if med_term in query_lower and (med_term in med_name or med_term in pactivos):
                return True
                
        # For cases where no explicit match, check if there's some partial overlap
        # but we'll be stricter to avoid false positives
        med_words = set(med_name.split() + pactivos.split())
        query_words = set(query_lower.split())
        common_words = med_words.intersection(query_words)
        if len(common_words) > 1 and not all(word in ["de", "la", "el", "los", "las", "para", "con", "sin"] for word in common_words):
            return True
            
        return False

    async def _get_complete_medication_details(self, session, med: Dict, fetch_prospecto: bool = False, focus_section: str = None) -> str:
        """
        Get comprehensive details for a medication with enhanced section focusing
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
        
        # 2. If we have a specific section to focus on, prioritize it
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
        
        # Reorder sections based on focus section
        if focus_section and focus_section in key_sections:
            # Move the focus section to the front
            focus_key = key_sections[focus_section]
            focused_sections = {focus_section: focus_key}
            for section, key in key_sections.items():
                if section != focus_section:
                    focused_sections[section] = key
            key_sections = focused_sections
        
        async def get_section(section, key):
            try:
                section_url = f"{self.base_url}/docSegmentado/contenido/1"
                async with session.get(section_url, params={"nregistro": nregistro, "seccion": section}) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, dict) and "contenido" in result:
                            content = result.get("contenido", "")
                            # For focus section, don't truncate
                            if section == focus_section:
                                return {"type": "section", "key": key, "data": content, "focused": True}
                            # For other sections, truncate as needed    
                            else:
                                if len(content) > 400:
                                    content = content[:397] + "..."
                                return {"type": "section", "key": key, "data": content, "focused": False}
            except Exception as e:
                logger.error(f"Error getting section {section}: {str(e)}")
            return {"type": "section", "key": key, "data": "No disponible", "focused": False}
        
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
        
        # Format into a token-efficient text description
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
            # Analyze the query to determine key elements
            query_info = self._analyze_query(message)
            is_prospecto = query_info["is_prospecto"]
            
            # Get relevant information from CIMA
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
            
            # Prepare the prompt with additional information about the query analysis
            prompt = f"""
Consulta: {message}

Análisis de la consulta:
- Principio activo detectado: {query_info['active_ingredient'] or 'No detectado'}
- Sección específica: {query_info['section_number'] or 'No especificada'}

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