"""
Improved FormulationAgent that uses the enhanced CIMA client
to generate more reliable medication formulations.
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
from improved_cima_client import CIMAClient
from search_graph import MedicationSearchGraph, QueryIntent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ImprovedFormulationAgent:
    """
    Improved agent for generating medication formulations using CIMA data.
    Uses the enhanced CIMAClient for better API integration.
    """
    openai_client: AsyncOpenAI
    cima_client: Optional[CIMAClient] = None
    reference_cache: Dict[str, List[Dict]] = field(default_factory=dict)
    base_url: str = Config.CIMA_BASE_URL
    max_tokens: int = 14000  # Leave room for prompt and response
    use_langgraph: bool = True  # Use the LangGraph search by default

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

    def __init__(self, openai_client: AsyncOpenAI):
        """Initialize the agent with OpenAI client and create a CIMA client"""
        self.openai_client = openai_client
        self.cima_client = CIMAClient(Config.CIMA_BASE_URL)
        self.reference_cache = {}
        self.base_url = Config.CIMA_BASE_URL
        self.max_tokens = 14000
        self.use_langgraph = True
        # Initialize tokenizer
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        # Active principle database - from the original agent
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

    def detect_formulation_type(self, query: str) -> Dict[str, Any]:
        """
        Enhanced formulation type detection with improved preposition handling
        
        Args:
            query: Query text
            
        Returns:
            Dictionary with detected formulation information
        """
        # Dictionary for formulation types and their keywords
        formulation_types = Config.FORMULATION_TYPES
        
        # Dictionary for pharmaceutical paths
        admin_routes = Config.ADMIN_ROUTES
        
        # Concentration patterns - extended to catch more variants
        concentration_pattern = r'(\d+(?:[,.]\d+)?\s*(?:%|mg|g|ml|mcg|UI|unidades)|\d+\s*(?:mg)?[/](?:ml|g))'
        
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
        
        # IMPROVED: Extract active principles with better preposition handling
        active_principle = None
        
        # First, clean the query by removing common prepositions to avoid misidentification
        cleaned_query = query_lower
        for prep in ['de', 'del', 'en', 'con', 'para', 'sobre', 'a', 'al']:
            cleaned_query = cleaned_query.replace(f' {prep} ', ' ')
        
        # Now search for active principles in the cleaned query
        for ap in self.active_principles:
            if ap in cleaned_query:
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
                    # IMPROVED: Better handling of common words to exclude
                    common_words = ['como', 'para', 'sobre', 'cual', 'este', 'esta', 'de', 'del', 'en', 'con']
                    words = [w for w in query_lower.split() if len(w) > 4 and w not in common_words]
                    if words:
                        active_principle = max(words, key=len)
        
        # Check for medication names like "MINOXIDIL BIORGA"
        med_names = re.findall(r'\b[A-Z]{2,}\s+[A-Z]{2,}\b', query.upper())
        if med_names:
            logger.info(f"Found medication names in all caps: {med_names}")
            # Add these to the active principle if not already found
            if not active_principle:
                active_principle = med_names[0].lower()
        
        return {
            "form_type": detected_form,
            "admin_route": detected_route,
            "concentration": concentration,
            "active_principle": active_principle,
            "is_prospecto": is_prospecto,
            "uppercase_names": med_names
        }

    async def format_medication_info(self, index: int, med: Dict, details: Dict, query_intent: Optional[QueryIntent] = None, is_critical: bool = False) -> str:
        """
        Format medication information for context
        
        Args:
            index: Reference index
            med: Basic medication data
            details: Detailed medication data
            query_intent: Query intent (optional)
            is_critical: Whether this is a critical reference
            
        Returns:
            Formatted information string
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
        lab_titular = basic_info.get('labtitular', med.get('labtitular', 'No disponible'))
        
        # Format sections with proper handling of missing data
        def get_section_content(section_key, max_len=1000):
            section_data = details.get(section_key, {})
            if not isinstance(section_data, dict):
                return "No disponible"
            
            content = section_data.get('contenido', 'No disponible')
            # Clean HTML tags for better readability
            if content and content != 'No disponible':
                # Simple HTML tag cleaning
                content = re.sub(r'<[^>]+>', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()
            
            # Use larger max_len for critical references
            max_length = 2000 if is_critical else max_len
            
            # Only truncate if really necessary
            if content and len(content) > max_length:
                return content[:max_length] + "... [Contenido truncado, ver ficha técnica completa]"
            return content
        
        # Handle special case for melatonina (not in CIMA)
        if nregistro == "custom_melatonina":
            return f"""
[Referencia {index}: Melatonina (Suplemento Dietético) (Nº Registro: custom_melatonina)]

NOTA IMPORTANTE: La melatonina se comercializa como suplemento alimenticio, no como medicamento registrado en CIMA.

INFORMACIÓN BÁSICA:
- Nombre: Melatonina (Suplemento Dietético)
- Tipo: Suplemento alimenticio (no medicamento)
- Principio activo: Melatonina

USOS HABITUALES:
- Ayuda para regular el ciclo del sueño
- Tratamiento de insomnio ocasional
- Regulación del ritmo circadiano

POSOLOGÍA HABITUAL:
- Generalmente entre 1mg y 5mg al día, tomado 30-60 minutos antes de acostarse.
- Consultar las indicaciones específicas del fabricante.

PRECAUCIONES:
- No es un medicamento. No debe utilizarse como sustituto de un tratamiento médico.
- Puede causar somnolencia.
- No recomendado durante el embarazo o la lactancia.

CONSIDERACIONES PARA FORMULACIÓN MAGISTRAL:
Aunque no es un medicamento registrado en CIMA, se puede preparar en cápsulas usando melatonina pura
de grado farmacéutico con excipientes adecuados como lactosa, almidón de maíz o celulosa microcristalina.
"""
        
        # If we have a specific query intent, prioritize that section
        if query_intent and query_intent.intent_type != "general" and query_intent.section_key:
            # Create a focused reference highlighting the requested information
            section_content = get_section_content(query_intent.section_key, 3000)  # Longer content for focused query
            
            reference = f"""
[Referencia {index}: {med_name} (Nº Registro: {nregistro})]

INFORMACIÓN SOBRE {query_intent.description.upper()} DE {med.get('pactivos', basic_info.get('pactivos', 'No disponible')).upper()}:

{section_content}

INFORMACIÓN BÁSICA:
- Nombre: {med_name}
- Número de registro: {nregistro}
- Laboratorio titular: {lab_titular}
- Principios activos: {med.get('pactivos', basic_info.get('pactivos', 'No disponible'))}

URL FICHA TÉCNICA:
https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html

URL PROSPECTO:
https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html
"""
            return reference
        
        # Default format for general queries, enhanced for critical references
        reference = f"""
[Referencia {index}: {med_name} (Nº Registro: {nregistro})]

INFORMACIÓN BÁSICA:
- Nombre: {med_name}
- Número de registro: {nregistro}
- Laboratorio titular: {lab_titular}
- Principios activos: {med.get('pactivos', basic_info.get('pactivos', 'No disponible'))}

COMPOSICIÓN:
{get_section_content('composicion')}

INDICACIONES TERAPÉUTICAS:
{get_section_content('indicaciones')}

POSOLOGÍA Y ADMINISTRACIÓN:
{get_section_content('posologia_procedimiento')}

CONTRAINDICACIONES:
{get_section_content('contraindicaciones')}
"""

        # For critical references, add more sections
        if is_critical:
            reference += f"""
ADVERTENCIAS Y PRECAUCIONES:
{get_section_content('advertencias')}

EXCIPIENTES:
{get_section_content('excipientes')}

CONSERVACIÓN:
{get_section_content('conservacion')}

INFORMACIÓN DEL PROSPECTO:
{get_section_content('prospecto_html', 2000) if 'prospecto_html' in details.get('prospecto', {}) else 'No disponible'}
"""
        else:
            # Non-critical references get fewer sections
            reference += f"""
ADVERTENCIAS Y PRECAUCIONES:
{get_section_content('advertencias')}

EXCIPIENTES:
{get_section_content('excipientes')}
"""
            
        # Check if important sections are all missing
        important_sections = ['composicion', 'indicaciones', 'posologia_procedimiento', 'contraindicaciones']
        all_unavailable = True
        for section in important_sections:
            content = get_section_content(section)
            if content and content != "No disponible" and not content.startswith("No disponible - Consultar"):
                all_unavailable = False
                break
        
        # Add note if all sections are missing
        if all_unavailable:
            reference += """

NOTA IMPORTANTE:
La información detallada de este medicamento no está completamente disponible en el formato estructurado de CIMA.
Se recomienda consultar directamente la ficha técnica completa a través de los enlaces proporcionados.
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
        Get relevant context for the query using the improved CIMA client
        
        Args:
            query: Query text
            n_results: Number of results to include
            
        Returns:
            Context text for formulation
        """
        logger.info(f"Getting context for query: '{query}'")
        cache_key = f"{query}_{n_results}"
        
        # Use cache if available
        if cache_key in self.reference_cache:
            cached_results = self.reference_cache[cache_key]
            
            # Extract query intent from cached data if available
            query_intent = None
            if len(cached_results) > 0 and len(cached_results[0]) > 2:
                query_intent = cached_results[0][2]  # Third element is query intent
                
            context_parts = [self.format_medication_info(i, med, details, query_intent, i==1) 
                             for i, (med, details, _) in enumerate(cached_results, 1)]
            
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

        # Extract formulation information for searching
        formulation_info = self.detect_formulation_type(query)
        active_principle = formulation_info.get("active_principle")
        uppercase_names = formulation_info.get("uppercase_names", [])
        
        # Create a MedicationSearchGraph instance for query intent analysis
        search_implementation = MedicationSearchGraph()
        results = []
        quality = "unknown"
        query_intent = None
        
        try:
            # First, analyze the query intent
            api_results, quality, query_intent = await search_implementation.execute_search(query)
            logger.info(f"Query intent: {query_intent.intent_type if query_intent else 'None'}, Quality: {quality}")
            
            # Special case for melatonina
            if active_principle and "melatonina" in active_principle.lower():
                logger.info("Detected query for melatonina (not in CIMA)")
                
                # Create custom result for melatonina
                med = {
                    "nregistro": "custom_melatonina",
                    "nombre": "Melatonina (Suplemento Dietético)",
                    "pactivos": "Melatonina",
                    "labtitular": "Varios fabricantes (suplemento)",
                    "comerc": True,
                }
                
                # Get melatonina custom data
                details = self.cima_client.get_melatonina_custom_data()
                
                # Store in cache and return
                self.reference_cache[cache_key] = [(med, details, query_intent)]
                
                return await self.format_medication_info(1, med, details, query_intent, True)
            
            # Check for uppercase medication names (like MINOXIDIL BIORGA)
            if uppercase_names:
                logger.info(f"Searching for uppercase medication name: {uppercase_names[0]}")
                
                # Try direct lookup through CIMA client
                direct_results = await self.cima_client.search_by_name(uppercase_names[0])
                
                if direct_results:
                    top_med = direct_results[0]
                    nregistro = top_med.get("nregistro")
                    
                    if nregistro:
                        # Get complete details using improved client
                        details = await self.cima_client.get_medication_details(nregistro)
                        
                        # Add to cache
                        self.reference_cache[cache_key] = [(top_med, details, query_intent)]
                        
                        # Return formatted info
                        return await self.format_medication_info(1, top_med, details, query_intent, True)
            
            # Search by active principle if available
            if active_principle:
                logger.info(f"Searching by active principle: {active_principle}")
                
                ap_results = await self.cima_client.search_by_active_principle(active_principle)
                
                if ap_results:
                    # Fetch details for top N results
                    cached_results = []
                    
                    for i, med in enumerate(ap_results[:n_results]):
                        nregistro = med.get("nregistro")
                        if nregistro:
                            details = await self.cima_client.get_medication_details(nregistro)
                            cached_results.append((med, details, query_intent))
                    
                    # Store in cache
                    if cached_results:
                        self.reference_cache[cache_key] = cached_results
                        
                        # Format results
                        context_parts = [
                            await self.format_medication_info(i, med, details, query_intent, i==1)
                            for i, (med, details, _) in enumerate(cached_results, 1)
                        ]
                        
                        # Check token count
                        full_context = "\n".join(context_parts)
                        if self.num_tokens(full_context) > self.max_tokens:
                            # Truncate as needed
                            smaller_context = "\n".join(context_parts[:2])
                            if self.num_tokens(smaller_context) > self.max_tokens:
                                return context_parts[0]
                            return smaller_context
                        
                        return full_context
            
            # If no results yet, try full query search
            if not results:
                logger.info(f"Trying full query search: {query}")
                
                name_results = await self.cima_client.search_by_name(query)
                
                if name_results:
                    # Fetch details for top results
                    cached_results = []
                    
                    for i, med in enumerate(name_results[:n_results]):
                        nregistro = med.get("nregistro")
                        if nregistro:
                            details = await self.cima_client.get_medication_details(nregistro)
                            cached_results.append((med, details, query_intent))
                    
                    # Store in cache and format
                    if cached_results:
                        self.reference_cache[cache_key] = cached_results
                        
                        # Format results
                        context_parts = [
                            await self.format_medication_info(i, med, details, query_intent, i==1)
                            for i, (med, details, _) in enumerate(cached_results, 1)
                        ]
                        
                        # Check token count
                        full_context = "\n".join(context_parts)
                        if self.num_tokens(full_context) > self.max_tokens:
                            smaller_context = "\n".join(context_parts[:2])
                            if self.num_tokens(smaller_context) > self.max_tokens:
                                return context_parts[0]
                            return smaller_context
                        
                        return full_context
        
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return (f"Error al buscar información en CIMA: {str(e)}. "
                   f"Por favor intente de nuevo o refine su consulta para ser más específica.")
        
        # If we reach this point, no results were found
        if "melatonina" in query.lower():
            # Return custom melatonina information
            med = {
                "nregistro": "custom_melatonina",
                "nombre": "Melatonina (Suplemento Dietético)",
                "pactivos": "Melatonina",
                "labtitular": "Varios fabricantes (suplemento)",
                "comerc": True,
            }
            
            details = self.cima_client.get_melatonina_custom_data()
            self.reference_cache[cache_key] = [(med, details, query_intent)]
            return await self.format_medication_info(1, med, details, query_intent, True)
        
        # No results found
        no_results_msg = (
            f"No se encontraron medicamentos relevantes en CIMA para esta consulta: '{query}'. "
            f"Por favor intente con términos más específicos o verifique que el principio activo "
            f"o medicamento esté registrado en CIMA."
        )
        
        if active_principle:
            no_results_msg += f"\n\nEl principio activo detectado fue: {active_principle}"
        
        return no_results_msg

    async def generate_response(self, query: str, context: str, query_intent: Optional[QueryIntent] = None) -> str:
        """
        Generate formulation response with GPT
        
        Args:
            query: Original query
            context: Context from CIMA
            query_intent: Query intent (optional)
            
        Returns:
            Generated formulation response
        """
        # Extract formulation details for improved prompting
        formulation_info = self.detect_formulation_type(query)
        
        # Select appropriate system prompt
        system_prompt = self.system_prompt
        
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
"""

        # Add intent information if available
        if query_intent and query_intent.intent_type != "general":
            prompt += f"""- Tipo de información solicitada: {query_intent.description}
- Sección de interés: {query_intent.section_key}
"""

        # Add the original query
        prompt += f"""
CONSULTA ORIGINAL:
{query}

Genera una respuesta completa y exhaustiva utilizando toda la información disponible en el contexto. Cita las fuentes CIMA usando el formato [Ref X: Nombre del medicamento (Nº Registro)].

Si no encuentras información suficiente en el contexto, indícalo claramente y sugiere consultar directamente la ficha técnica a través de los enlaces proporcionados.
"""

        # Special case for melatonina
        if formulation_info["active_principle"] and "melatonina" in formulation_info["active_principle"].lower():
            prompt += """
IMPORTANTE: La melatonina se comercializa principalmente como suplemento alimenticio, no como medicamento registrado en CIMA.
Ten en cuenta este factor al elaborar la formulación, basándote en los datos disponibles de suplementos y en principios generales
de formulación magistral para cápsulas, suspensiones u otras formas farmacéuticas solicitadas.
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
        """
        Process a question and generate a formulation answer
        
        Args:
            question: Question about medication formulation
            
        Returns:
            Dictionary with answer, context and references
        """
        # Get context for the question
        context = await self.get_relevant_context(question)
        
        # Extract query intent to pass to the generator
        search_implementation = MedicationSearchGraph()
        _, _, query_intent = await search_implementation.execute_search(question)
        
        # Generate response
        answer = await self.generate_response(question, context, query_intent)
        
        # Process links in response
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
        """Close resources properly"""
        if self.cima_client:
            await self.cima_client.close()
