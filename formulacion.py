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

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a text string for a specific model."""
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception as e:
        # If there's an error, estimate with a simple approximation
        # (roughly 4 characters per token for most languages)
        return len(text) // 4

@dataclass
class FormulationAgent:
    openai_client: AsyncOpenAI
    base_url: str = Config.CIMA_BASE_URL
    reference_cache: Dict[str, List[Dict]] = field(default_factory=dict)
    session: aiohttp.ClientSession = None
    max_tokens: int = 12000  # Conservative limit to leave room for response

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

    async def get_session(self):
        """Get or create an aiohttp session with keepalive"""
        if self.session is None or self.session.closed:
            # Using TCPConnector with keepalive and increased limits
            connector = aiohttp.TCPConnector(
                ssl=False, 
                limit=30,  # Increased connection limit
                keepalive_timeout=120  # Longer keepalive
            )
            timeout = aiohttp.ClientTimeout(total=60)  # Increased timeout to 60 seconds
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self.session
    
    async def get_medication_details(self, nregistro: str) -> Dict:
        """Optimized method to fetch medication details focusing on essential data"""
        logger.info(f"Fetching medication details for nregistro: {nregistro}")
        session = await self.get_session()
        
        # Prioritize the most important sections for formulación magistral
        essential_sections = {
            "2": "composicion",
            "4.1": "indicaciones",
            "4.2": "posologia_procedimiento",
            "4.3": "contraindicaciones",
            "4.4": "advertencias",
            "6.1": "excipientes",
            "6.3": "conservacion"
        }
        
        details = {}
        
        # Define async tasks for concurrent execution - focusing on basic info first
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
        
        # Execute basic info request first - we need this for sure
        basic_info = await get_basic_info()
        details["basic"] = basic_info
        
        # Only fetch sections if we got basic info successfully and it's not an error
        if "error" not in basic_info:
            # Limit concurrent section requests to avoid overwhelming the API
            semaphore = asyncio.Semaphore(5)  # Reduced to 5 concurrent requests
            
            async def get_section(section, key):
                tech_url = f"{self.base_url}/docSegmentado/contenido/1"
                params = {"nregistro": nregistro, "seccion": section}
                try:
                    async with semaphore, session.get(tech_url, params=params) as response:
                        if response.status == 200:
                            result = await response.json()
                            if isinstance(result, dict):
                                return (key, result)
                except Exception as e:
                    logger.error(f"Error retrieving section {section}: {str(e)}")
                return (key, {"contenido": f"No disponible"})
            
            # Create tasks only for essential sections to reduce data volume
            section_tasks = [get_section(section, key) for section, key in essential_sections.items()]
            
            # Execute section tasks concurrently and process results
            section_results = await asyncio.gather(*section_tasks)
            for key, value in section_results:
                details[key] = value
        
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
        Optimized medication information formatting focused on essential data
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
        
        # Format sections with proper handling of missing data
        def get_section_content(section_key, max_length=400):
            """Get section content with length limit"""
            section_data = details.get(section_key, {})
            if not isinstance(section_data, dict):
                return "No disponible"
            content = section_data.get('contenido', 'No disponible')
            # Limit length to reduce token usage
            if len(content) > max_length:
                return content[:max_length] + "..."
            return content
        
        # Create a concise reference focused on formulary-relevant information
        reference = f"""
[Referencia {index}: {med_name} (Nº Registro: {nregistro})]

INFORMACIÓN BÁSICA:
- Nombre: {med_name}
- Número de registro: {nregistro}
- Laboratorio titular: {lab_titular}
- Principios activos: {med.get('pactivos', 'No disponible')}

COMPOSICIÓN:
{get_section_content('composicion')}

EXCIPIENTES:
{get_section_content('excipientes')}

INDICACIONES TERAPÉUTICAS:
{get_section_content('indicaciones')}

POSOLOGÍA Y ADMINISTRACIÓN:
{get_section_content('posologia_procedimiento')}

ADVERTENCIAS Y PRECAUCIONES:
{get_section_content('advertencias')}

CONSERVACIÓN:
{get_section_content('conservacion')}

URL FICHA TÉCNICA:
https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html
"""
        return reference

    async def get_relevant_context(self, query: str, n_results: int = 3) -> str:
        """
        Enhanced context retrieval that prioritizes most relevant information
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
            context = [self.format_medication_info(i, med, details) 
                      for i, (med, details) in enumerate(cached_results, 1)]
            joined_context = "\n".join(context)
            
            # Check if we need to truncate to stay within token limits
            token_count = count_tokens(joined_context + self.system_prompt, Config.CHAT_MODEL)
            if token_count > self.max_tokens:
                logger.warning(f"Context too large ({token_count} tokens), truncating...")
                return self.truncate_context(joined_context)
            
            return joined_context

        # Get/create aiohttp session
        session = await self.get_session()
        search_url = f"{self.base_url}/medicamentos"
        
        # Define search strategies prioritizing exact matches
        search_strategies = [
            # Try exact match first
            {"params": {"nombre": query}, "priority": 1},
            # Search by active principle
            {"params": {"principiosActivos": active_principle}, "priority": 2},
            # Search by pharmaceutical form
            {"params": {"formaFarmaceutica": formulation_type}, "priority": 3},
        ]
        
        # Add any additional terms that might be relevant (limit to 2 for performance)
        terms = self._extract_search_terms(query)[:2]
        for term in terms:
            if term != query and term != active_principle:
                search_strategies.append({
                    "params": {"nombre": term}, 
                    "priority": 4
                })
        
        # Perform searches concurrently
        async def search_medications(params, priority):
            try:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, dict) and "resultados" in data:
                            results = data.get("resultados", [])
                            logger.info(f"Search with params {params} returned {len(results)} results")
                            return {"results": results, "priority": priority}
            except Exception as e:
                logger.error(f"Error in search with params {params}: {str(e)}")
            return {"results": [], "priority": priority}
        
        # Execute searches concurrently
        search_tasks = [search_medications(strategy["params"], strategy["priority"]) 
                         for strategy in search_strategies]
        
        search_results = await asyncio.gather(*search_tasks)
        
        # Combine results with prioritization
        all_results = []
        seen_nregistros = set()
        
        # First add results from high priority searches
        for priority in range(1, 5):  # Process in priority order
            for result_set in search_results:
                if result_set["priority"] == priority:
                    for med in result_set["results"]:
                        if isinstance(med, dict) and med.get("nregistro"):
                            nregistro = med.get("nregistro")
                            if nregistro not in seen_nregistros:
                                seen_nregistros.add(nregistro)
                                all_results.append(med)
        
        # Limit results for better performance (use original n_results parameter)
        results = all_results[:n_results]
        cached_results = []

        if results:
            logger.info(f"Found {len(results)} relevant medications, fetching details")
            
            # Fetch details for all medications concurrently
            semaphore = asyncio.Semaphore(3)  # Lower to 3 concurrent detail fetches
            
            async def fetch_med_details(med):
                async with semaphore:
                    if not isinstance(med, dict) or not med.get("nregistro"):
                        return None
                    details = await self.get_medication_details(med["nregistro"])
                    return (med, details)
            
            # Create tasks for fetching medication details
            detail_tasks = [fetch_med_details(med) for med in results]
            detail_results = await asyncio.gather(*detail_tasks)
            
            # Filter out None results
            cached_results = [result for result in detail_results if result is not None]
            
            # Store in cache
            self.reference_cache[cache_key] = cached_results
            
            # Format context with complete information
            context = [self.format_medication_info(i, med, details) 
                      for i, (med, details) in enumerate(cached_results, 1)]
            
            joined_context = "\n".join(context)
            
            # Check if we need to truncate to stay within token limits
            token_count = count_tokens(joined_context + self.system_prompt, Config.CHAT_MODEL)
            if token_count > self.max_tokens:
                logger.warning(f"Context too large ({token_count} tokens), truncating...")
                return self.truncate_context(joined_context)
            
            return joined_context
        
        return "No se encontraron resultados relevantes."
    
    def truncate_context(self, context: str) -> str:
        """Truncate context to fit within token limits"""
        system_tokens = count_tokens(self.system_prompt, Config.CHAT_MODEL)
        max_context_tokens = self.max_tokens - system_tokens - 500  # Leave 500 tokens buffer
        
        # First try: reduce each reference section separately
        references = context.split("[Referencia ")
        
        # Keep the first reference (header)
        if not references:
            return "No se encontraron referencias relevantes."
        
        # Process each reference to make it more concise
        processed_refs = []
        current_tokens = 0
        
        # Start with the header if it exists
        header = references[0]
        processed_refs.append(header)
        current_tokens += count_tokens(header, Config.CHAT_MODEL)
        
        # Process each reference section
        for i, ref in enumerate(references[1:], 1):
            # Restore the reference prefix
            ref = "[Referencia " + ref
            
            # Check if adding this reference would exceed the token limit
            ref_tokens = count_tokens(ref, Config.CHAT_MODEL)
            
            if current_tokens + ref_tokens <= max_context_tokens:
                # If it fits, add the whole reference
                processed_refs.append(ref)
                current_tokens += ref_tokens
            else:
                # We need to truncate this reference
                # Keep only the essential parts (basic info, composition, indications)
                lines = ref.split('\n')
                essential_ref = []
                in_essential_section = False
                
                for line in lines:
                    if "INFORMACIÓN BÁSICA:" in line or "COMPOSICIÓN:" in line or "EXCIPIENTES:" in line:
                        in_essential_section = True
                    elif "ADVERTENCIAS" in line or "CONSERVACIÓN" in line:
                        in_essential_section = False
                    
                    if in_essential_section or "URL FICHA TÉCNICA:" in line or "[Referencia" in line:
                        essential_ref.append(line)
                
                truncated_ref = '\n'.join(essential_ref)
                truncated_tokens = count_tokens(truncated_ref, Config.CHAT_MODEL)
                
                if current_tokens + truncated_tokens <= max_context_tokens:
                    processed_refs.append(truncated_ref)
                    current_tokens += truncated_tokens
                else:
                    # Can't even fit essential parts, we'll stop here
                    break
        
        return '\n'.join(processed_refs)

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
        
        # Add individual words that might be medication names (skip common words)
        common_words = {"sobre", "para", "como", "este", "esta", "estos", "estas", "cual", "cuales", 
                        "con", "por", "los", "las", "del", "que", "realizar", "realizar", "redactar", 
                        "crear", "generar", "prospecto", "formular", "elaborar"}
        
        words = query.split()
        for word in words:
            if len(word) > 4 and word.lower() not in common_words and word not in potential_terms:
                potential_terms.append(word)
        
        # Add bi-grams (pairs of words) - limit to first 2 words
        if len(words) > 1:
            if len(words[0]) > 3 and len(words[1]) > 3:
                bigram = f"{words[0]} {words[1]}"
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

        # Check if the prompt exceeds token limit
        token_count = count_tokens(prompt + system_prompt, Config.CHAT_MODEL)
        if token_count > self.max_tokens:
            logger.warning(f"Prompt too large ({token_count} tokens), truncating context...")
            # Truncate context to fit within limits
            max_context_tokens = self.max_tokens - count_tokens(system_prompt + prompt.replace("{context}", ""), Config.CHAT_MODEL)
            context_tokens = count_tokens(context, Config.CHAT_MODEL)
            
            if context_tokens > max_context_tokens:
                # Truncate context to fit within limits
                context = self.truncate_context(context)
                # Rebuild prompt with truncated context
                prompt = prompt.replace("{context}", context)

        logger.info("Generating response with OpenAI")
        response = await self.openai_client.chat.completions.create(
            model=Config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content

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
        """Close the aiohttp session to free resources"""
        if self.session and not self.session.closed:
            await self.session.close()

@dataclass
class CIMAExpertAgent:
    openai_client: AsyncOpenAI
    reference_cache: Dict[str, str] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    base_url: str = Config.CIMA_BASE_URL
    session: aiohttp.ClientSession = None
    max_tokens: int = 12000  # Conservative limit to leave room for response
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.reference_cache = {}
        self.base_url = Config.CIMA_BASE_URL
        self.conversation_history = []
        self.session = None

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
            # Using TCPConnector with keepalive and increased limits
            connector = aiohttp.TCPConnector(
                ssl=False, 
                limit=30,  # Increased connection limit
                keepalive_timeout=120  # Longer keepalive
            )
            timeout = aiohttp.ClientTimeout(total=60)  # Increased timeout to 60 seconds
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self.session

    async def get_medication_info(self, query: str) -> str:
        """
        Optimized implementation for obtaining medication information
        """
        cache_key = f"query_{query}"
        if cache_key in self.reference_cache:
            cached_result = self.reference_cache[cache_key]
            # Check if cached result fits token limit
            if count_tokens(cached_result + self.system_prompt, Config.CHAT_MODEL) <= self.max_tokens:
                return cached_result
            else:
                # Truncate if needed
                return self.truncate_context(cached_result)

        # Detect if this is a prospecto request
        prospecto_pattern = r'(?:redactar|generar|crear|elaborar|realizar?e?|escrib[ei]r|hac[ae]r|desarroll[ae]r)\s+(?:un|el|uns?|una?)?\s+prospecto'
        is_prospecto = bool(re.search(prospecto_pattern, query.lower()))
        logger.info(f"Query: '{query}', Is prospecto: {is_prospecto}")
        
        # Extract potential terms for search
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
        
        # Optimize search strategy: perform direct search first, then iterate through terms
        # First try with full query if it's not a prospecto request
        if not is_prospecto:
            meds = await self._search_medications(session, query)
            for med in meds[:3]:  # Limit to top 3 results for performance
                if med.get("nregistro") not in processed_nregistros:
                    processed_nregistros.add(med.get("nregistro"))
                    med_info = await self._get_optimized_medication_details(session, med)
                    if med_info:
                        all_med_info.append(med_info)
        
        # For prospecto requests or if we didn't get results, focus on the extracted terms
        if is_prospecto or len(all_med_info) < 2:  # Need at least 2 results for good context
            for term in potential_terms[:3]:  # Limit to top 3 terms for performance
                term_meds = await self._search_medications(session, term)
                for med in term_meds[:2]:  # Max 2 per term
                    if med.get("nregistro") not in processed_nregistros:
                        processed_nregistros.add(med.get("nregistro"))
                        med_info = await self._get_optimized_medication_details(session, med, fetch_prospecto=is_prospecto)
                        if med_info:
                            all_med_info.append(med_info)
                            
                # Limit total results to manage context size
                if len(all_med_info) >= 3:
                    break
        
        # Combine all results
        combined_results = "\n\n".join(all_med_info)
        
        # Check token count and truncate if needed
        token_count = count_tokens(combined_results + self.system_prompt, Config.CHAT_MODEL)
        if token_count > self.max_tokens:
            logger.warning(f"Context too large ({token_count} tokens), truncating...")
            combined_results = self.truncate_context(combined_results)
        
        # Store in cache
        self.reference_cache[cache_key] = combined_results
        return combined_results

    def _extract_search_terms(self, query: str) -> List[str]:
        """
        Extract potentially relevant search terms from the query
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
            if i < 2 and len(words[i]) > 3 and len(words[i+1]) > 3:  # Limit to first 2 pairs
                bigram = f"{words[i]} {words[i+1]}"
                potential_terms.append(bigram)
        
        # Remove duplicates
        seen = set()
        return [x for x in potential_terms if x.lower() not in seen and not seen.add(x.lower())][:5]  # Limit to top 5

    async def _search_medications(self, session, query: str) -> List[Dict]:
        """
        Search medications using multiple strategies
        """
        search_url = f"{self.base_url}/medicamentos"
        all_results = []
        
        # Search strategies in order of relevance - prioritize specific searches
        search_strategies = [
            {"params": {"nombre": query}},
            {"params": {"practiv1": query}}
        ]
        
        # Execute first two strategies concurrently for speed
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
        
        # Run primary searches concurrently
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
        Search in technical files for more comprehensive results (used as fallback)
        """
        results = []
        search_url = f"{self.base_url}/buscarEnFichaTecnica"
        
        # Important sections in ficha técnica for comprehensive search (reduced to key sections)
        sections = ["4.1", "4.2"]
        
        # Extract key words for search (limit to 2 words for performance)
        words = [word for word in query.split() if len(word) > 3][:2]
        if not words:
            return []
        
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
        
        return results[:3]  # Limit to top 3 results

    async def _get_optimized_medication_details(self, session, med: Dict, fetch_prospecto: bool = False) -> str:
        """
        Get essential medication details in an optimized way
        """
        if not isinstance(med, dict) or not med.get("nregistro"):
            return ""
        
        nregistro = med.get("nregistro")
        
        # Define the essential sections to retrieve (focusing on the most important ones)
        essential_sections = {
            "4.1": "indicaciones",
            "4.2": "posologia",
            "4.3": "contraindicaciones",
            "4.8": "efectos_adversos"
        }
        
        # 1. Get basic information first (most important)
        basic_info = {}
        try:
            url = f"{self.base_url}/medicamento"
            async with session.get(url, params={"nregistro": nregistro}) as response:
                if response.status == 200:
                    result = await response.json()
                    if isinstance(result, dict):
                        basic_info = result
        except Exception as e:
            logger.error(f"Error getting basic info: {str(e)}")
            return ""  # If we can't get basic info, abort
        
        # 2. Get essential sections concurrently
        sections_data = {}
        
        # Limit concurrent requests
        semaphore = asyncio.Semaphore(4)
        
        async def fetch_section(section, key):
            async with semaphore:
                try:
                    section_url = f"{self.base_url}/docSegmentado/contenido/1"
                    async with session.get(section_url, params={"nregistro": nregistro, "seccion": section}) as response:
                        if response.status == 200:
                            result = await response.json()
                            if isinstance(result, dict) and "contenido" in result:
                                content = result.get("contenido", "")
                                # Limit content length to reduce token usage (300 chars per section)
                                if len(content) > 300:
                                    content = content[:297] + "..."
                                return (key, content)
                except Exception as e:
                    logger.error(f"Error getting section {section}: {str(e)}")
                return (key, "No disponible")
        
        # Execute section fetches concurrently
        section_tasks = [fetch_section(section, key) for section, key in essential_sections.items()]
        section_results = await asyncio.gather(*section_tasks)
        
        for key, content in section_results:
            sections_data[key] = content
        
        # 3. Get prospecto if requested (only essential for prospecto requests)
        prospecto_data = None
        if fetch_prospecto:
            try:
                # Try the direct HTML endpoint first (more efficient)
                prospecto_url = f"https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html"
                async with session.get(prospecto_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Extract just the main content to reduce size
                        # Simple approximation - extract content between main tags if present
                        main_content = re.search(r'<main.*?>(.*?)</main>', content, re.DOTALL)
                        if main_content:
                            prospecto_data = main_content.group(1)
                        else:
                            # If no main tags, just take a portion of the content
                            prospecto_data = content[:1000] + "..."
            except Exception as e:
                logger.error(f"Error getting prospecto: {str(e)}")
        
        # 4. Format the data into a concise text representation
        # Basic details
        name = med.get("nombre", basic_info.get("nombre", "No disponible"))
        pactivos = med.get("pactivos", basic_info.get("pactivos", "No disponible"))
        lab = basic_info.get("labtitular", "No disponible")
        
        # URLs
        ficha_url = f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html"
        prospecto_url = f"https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html"
        
        # Format dates if available
        estado = basic_info.get("estado", {})
        fecha_aut = self._format_date(estado.get("aut", "")) if isinstance(estado, dict) else "No disponible"
        
        # Build the information block - focus on essentials
        info_parts = [
            f"[Ref: {name} (Nº Registro: {nregistro})]",
            f"Nombre: {name}",
            f"Principios activos: {pactivos}",
            f"Laboratorio: {lab}",
            f"Fecha autorización: {fecha_aut}"
        ]
        
        # Add essential sections
        for key, title in {
            "indicaciones": "INDICACIONES TERAPÉUTICAS",
            "posologia": "POSOLOGÍA Y FORMA DE ADMINISTRACIÓN",
            "contraindicaciones": "CONTRAINDICACIONES",
            "efectos_adversos": "EFECTOS ADVERSOS"
        }.items():
            content = sections_data.get(key, "")
            if content and len(content.strip()) > 0:
                info_parts.append(f"{title}:\n{content}")
        
        # Add prospecto data if available (for prospecto requests)
        if prospecto_data:
            # Clean HTML tags for readability
            clean_prospecto = re.sub(r'<[^>]+>', ' ', prospecto_data)
            clean_prospecto = re.sub(r'\s+', ' ', clean_prospecto).strip()
            if len(clean_prospecto) > 500:  # Ensure it's not too long
                clean_prospecto = clean_prospecto[:497] + "..."
            info_parts.append(f"PROSPECTO (extracto):\n{clean_prospecto}")
        
        # Add links to technical data sheet and prospecto
        info_parts.append(f"Ficha técnica completa: {ficha_url}")
        info_parts.append(f"Prospecto completo: {prospecto_url}")
        
        return "\n\n".join(info_parts)

    def truncate_context(self, context: str) -> str:
        """Truncate context to fit within token limits"""
        # Calculate how many tokens we can use for context
        system_tokens = count_tokens(self.system_prompt, Config.CHAT_MODEL)
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history[-4:]])
        history_tokens = count_tokens(history_text, Config.CHAT_MODEL)
        
        # Leave room for the response (4000 tokens) and user query (500 tokens)
        available_tokens = self.max_tokens - system_tokens - history_tokens - 4500
        
        # If available tokens is negative, we need to truncate history instead
        if available_tokens < 1000:  # Minimum threshold
            # Reduce history usage
            self.conversation_history = self.conversation_history[-2:]
            # Recalculate
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history])
            history_tokens = count_tokens(history_text, Config.CHAT_MODEL)
            available_tokens = self.max_tokens - system_tokens - history_tokens - 4500
        
        # Ensure we have at least 1000 tokens for context
        available_tokens = max(1000, available_tokens)
        
        # If context is already within limits, return as is
        context_tokens = count_tokens(context, Config.CHAT_MODEL)
        if context_tokens <= available_tokens:
            return context
        
        # Split context into medication references
        references = context.split("[Ref:")
        if not references:
            return "No se encontró información relevante."
        
        # Start with any header text
        result = references[0]
        current_tokens = count_tokens(result, Config.CHAT_MODEL)
        
        # Process each reference
        for i, ref in enumerate(references[1:], 1):
            # Restore the reference prefix
            ref_text = "[Ref:" + ref
            ref_tokens = count_tokens(ref_text, Config.CHAT_MODEL)
            
            # If adding this reference would exceed the limit
            if current_tokens + ref_tokens > available_tokens:
                # If this is the first reference, we need to include at least part of it
                if i == 1:
                    # Extract the essential parts only
                    ref_lines = ref_text.split('\n')
                    essential_lines = []
                    
                    # Include reference header and basic info
                    for j, line in enumerate(ref_lines):
                        if j < 5 or "INDICACIONES" in line:  # First 5 lines + indications
                            essential_lines.append(line)
                    
                    # Add URLs at the end
                    for line in ref_lines:
                        if "Ficha técnica" in line or "Prospecto" in line:
                            essential_lines.append(line)
                    
                    partial_ref = '\n'.join(essential_lines)
                    partial_tokens = count_tokens(partial_ref, Config.CHAT_MODEL)
                    
                    if current_tokens + partial_tokens <= available_tokens:
                        result += partial_ref
                    else:
                        # We're really out of space, just take the first few lines
                        first_lines = '\n'.join(ref_lines[:3])
                        result += first_lines + "\n[...información truncada por limitaciones de espacio...]"
                
                # Add a note that information was truncated
                result += "\n\n[Se ha truncado parte de la información debido a limitaciones de tamaño.]"
                break
            else:
                # This reference fits, add it
                result += ref_text
                current_tokens += ref_tokens
        
        return result

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
        Handle chat messages with comprehensive context but within token limits
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
            
            # Calculate token usage to ensure we're within limits
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history[-4:]])
            message_tokens = count_tokens(message, Config.CHAT_MODEL)
            system_tokens = count_tokens(system_prompt, Config.CHAT_MODEL)
            history_tokens = count_tokens(history_text, Config.CHAT_MODEL)
            context_tokens = count_tokens(context, Config.CHAT_MODEL)
            
            total_tokens = system_tokens + history_tokens + message_tokens + context_tokens
            logger.info(f"Token usage estimate: {total_tokens} (limit: {self.max_tokens})")
            
            # If we're over the limit, truncate the context
            if total_tokens > self.max_tokens:
                logger.warning(f"Total tokens ({total_tokens}) exceeds limit, truncating context")
                context = self.truncate_context(context)
                # Recalculate
                context_tokens = count_tokens(context, Config.CHAT_MODEL)
                total_tokens = system_tokens + history_tokens + message_tokens + context_tokens
                logger.info(f"After truncation: {total_tokens} tokens")
            
            # Prepare the prompt
            prompt = f"""
Consulta: {message}

Contexto relevante de CIMA:
{context if context else "No se encontró información específica en CIMA para esta consulta."}

Responde de manera detallada y precisa, citando las fuentes específicas del contexto.
{"Genera un prospecto completo siguiendo las directrices de la AEMPS." if is_prospecto else ""}
"""
            
            # Using last 4 messages for context 
            recent_history = self.conversation_history[-4:] if len(self.conversation_history) > 4 else self.conversation_history
            
            messages = [
                {"role": "system", "content": system_prompt},
                *recent_history,
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

            return {
                "answer": assistant_response_with_links,
                "context": context,
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
        """Close the aiohttp session to free resources"""
        if self.session and not self.session.closed:
            await self.session.close()