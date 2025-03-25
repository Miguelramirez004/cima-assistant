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
class FormulationAgent:
    openai_client: AsyncOpenAI
    base_url: str = Config.CIMA_BASE_URL
    reference_cache: Dict[str, List[Dict]] = field(default_factory=dict)
    session: aiohttp.ClientSession = None
    max_tokens: int = 14000  # Leave room for prompt and response
    use_langgraph: bool = True  # Use the new LangGraph search by default

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

    focused_information_prompt = """Experto en información farmacéutica con amplia experiencia en consultas a CIMA.

Has recibido una consulta específica sobre {information_type} de {active_principle}. Proporciona una respuesta directa, clara y detallada centrada específicamente en esta consulta.

Tu respuesta debe:
1. Comenzar con un resumen conciso de la información solicitada
2. Proporcionar todos los detalles relevantes sobre {information_type} del medicamento
3. Incluir referencias específicas a las fuentes CIMA utilizadas, con el formato [Ref X: Nombre del medicamento (Nº Registro)]
4. Estar estructurada de forma clara con subtítulos si es necesario
5. Priorizar la información oficial disponible en CIMA

Si hay distintas presentaciones o formas farmacéuticas del medicamento, menciona las diferencias relevantes entre ellas. Si hay información contradictoria, indícalo claramente y justifica qué fuente consideras más fiable.

Utiliza un lenguaje preciso pero accesible, recordando que la persona que consulta puede ser un profesional sanitario o un paciente.
"""

    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.reference_cache = {}
        self.base_url = Config.CIMA_BASE_URL
        self.session = None
        self.max_tokens = 14000
        self.use_langgraph = True  # Default to using improved search
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
"minoxidil", "nolotil", "escitalopram", "bromazepam", "pantoprazol",
"citalopram", "esomeprazol", "sertralina", "bisoprolol", "olmesartan",
"rosuvastatina", "duloxetina", "clopidogrel", "furosemida", "ramipril",
"paroxetina", "micofenolato", "olanzapina", "lansoprazol", "irbesartan",
"nebivolol", "torasemida", "pregabalina", "venlafaxina", "gabapentina",
"carvedilol", "tamsulosina", "telmisartan", "metoclopramida", "levocetirizina",
"dexketoprofeno", "deflazacort", "mirtazapina", "ebastina", "propranolol",
"candesartan", "sildenafilo", "tacrolimus", "amlodipino/valsartan", "ezetimiba",
"levonorgestrel", "raltegravir", "donepezilo", "fexofenadina", "clortalidona",
"trazodona", "levetiracetam", "solifenacina", "rivaroxaban", "glimepirida",
"memantina"
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
    
    async def get_medication_details(self, nregistro: str) -> Dict:
        """Enhanced method to fetch medication details with better error handling and fallbacks"""
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
        errors = []
        
        # Define async tasks - with enhanced retry logic
        async def get_basic_info():
            detail_url = f"{self.base_url}/medicamento"
            retry_count = 3
            
            for attempt in range(retry_count):
                try:
                    async with session.get(detail_url, params={"nregistro": nregistro}) as response:
                        if response.status == 200:
                            try:
                                # Try to parse as JSON first
                                result = await response.json()
                                if isinstance(result, dict):
                                    return result
                            except Exception as e:
                                logger.warning(f"Error parsing JSON response for basic info (attempt {attempt+1}): {str(e)}")
                                # Try as text if JSON fails
                                text = await response.text()
                                if text and len(text) > 50:
                                    # Try to parse as JSON again with some preprocessing
                                    text = text.strip()
                                    if text.startswith('{') and text.endswith('}'):
                                        try:
                                            import json
                                            result = json.loads(text)
                                            if isinstance(result, dict):
                                                return result
                                        except:
                                            pass
                                
                                logger.info(f"Retrieved text response for basic info (len: {len(text)})")
                                return {"nombre": med_name_from_nregistro(nregistro)}
                        else:
                            logger.warning(f"Non-200 status for basic info (attempt {attempt+1}): {response.status}")
                            
                    # Exponential backoff between retries
                    if attempt < retry_count - 1:
                        await asyncio.sleep(1 * (attempt + 1))
                except Exception as e:
                    logger.error(f"Error retrieving basic details (attempt {attempt+1}): {str(e)}")
                    if attempt < retry_count - 1:
                        await asyncio.sleep(1 * (attempt + 1))
            
            errors.append(f"Error obteniendo información básica después de {retry_count} intentos")
            return {
                "nombre": med_name_from_nregistro(nregistro),
                "nregistro": nregistro,
                "error": "Unable to retrieve basic details"
            }
        
        # Helper function to guess medication name from registration number if all else fails
        def med_name_from_nregistro(nreg):
            return f"Medicamento (Nº Registro: {nreg})"
        
        # Enhanced section retrieval with better fallbacks
        async def get_section(section, key):
            # First attempt: Direct API call
            tech_url = f"{self.base_url}/docSegmentado/contenido/1"
            params = {"nregistro": nregistro, "seccion": section}
            
            for attempt in range(3):  # Try up to 3 times
                try:
                    async with session.get(tech_url, params=params) as response:
                        if response.status == 200:
                            try:
                                # Try to parse as JSON first
                                result = await response.json()
                                if isinstance(result, dict) and result.get("contenido") and result.get("contenido") != "No disponible":
                                    return (key, result)
                            except Exception as e:
                                # If JSON parsing fails, try as text
                                logger.warning(f"JSON parsing failed for section {section}: {str(e)}")
                                text = await response.text()
                                if text and len(text) > 50:  # Basic validity check
                                    return (key, {"contenido": text})
                        else:
                            logger.warning(f"Non-200 status for section {section} (attempt {attempt+1}): {response.status}")
                    
                    # Only sleep and retry if all attempts haven't been exhausted
                    if attempt < 2:
                        await asyncio.sleep(1 * (attempt + 1))
                except Exception as e:
                    logger.warning(f"Error in attempt {attempt+1} for section {section}: {str(e)}")
                    if attempt < 2:
                        await asyncio.sleep(1 * (attempt + 1))
            
            # More aggressive fallback: Try a variety of URLs and patterns
            try:
                # List of potential URLs to try
                potential_urls = [
                    f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/{section}/FichaTecnica.html",
                    f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html",
                    f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FichaTecnica_{nregistro}.html",
                    f"https://cima.aemps.es/cima/pdfs/ft/{nregistro}/ft_{nregistro}.pdf",
                    f"https://cima.aemps.es/cima/rest/medicamento?nregistro={nregistro}"
                ]
                
                # Patterns to try extracting sections from full documents
                section_patterns = [
                    f'<h3[^>]*>{section}\\.[^<]+</h3>(.*?)(?:<h3|<div class="section-break">|<div class="section">)',
                    f'<h[1-6][^>]*>{section}\\.[^<]+</h[1-6]>(.*?)(?:<h[1-6]|<div)',
                    f'{section}\\.[^<\\n]+(?:<br[^>]*>|\\n)(.*?)(?:{section}\\.\\d+|<h[1-6]|<div)',
                    f'"{section}"\\s*:\\s*"([^"]+)"',  # JSON pattern
                    f'"{section}"\\s*:\\s*\\[(.*?)\\]'   # JSON array pattern
                ]
                
                # Try each URL
                for url in potential_urls:
                    try:
                        logger.info(f"Trying fallback URL: {url}")
                        async with session.get(url) as response:
                            if response.status == 200:
                                content = await response.text()
                                if content and len(content) > 100:
                                    # Try to extract the section using patterns
                                    for pattern in section_patterns:
                                        try:
                                            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                                            if match:
                                                section_content = match.group(1).strip()
                                                if len(section_content) > 50:
                                                    logger.info(f"Found section {section} using pattern in {url}")
                                                    return (key, {"contenido": section_content})
                                        except Exception as e:
                                            logger.warning(f"Error with pattern extraction: {str(e)}")
                                    
                                    # If this is the basic medicamento info, try to extract it as a JSON object
                                    if url.endswith(f"nregistro={nregistro}"):
                                        try:
                                            import json
                                            data = json.loads(content)
                                            # Look for relevant keys based on the section
                                            section_mapping = {
                                                "2": "composicion",
                                                "4.1": "indicaciones",
                                                "4.3": "contraindicaciones"
                                            }
                                            
                                            if section in section_mapping and section_mapping[section] in data:
                                                return (key, {"contenido": data[section_mapping[section]]})
                                        except:
                                            pass
                    except Exception as e:
                        logger.warning(f"Error accessing fallback URL {url}: {str(e)}")
            except Exception as e:
                logger.warning(f"Error in fallback mechanism for {section}: {str(e)}")
            
            # If all attempts fail, return placeholder with link to full document
            logger.warning(f"All attempts failed for section {section}")
            errors.append(f"No se pudo obtener la sección {section}")
            return (key, {
                "contenido": f"No disponible - Consultar la ficha técnica completa en: https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html",
                "error": f"Failed to retrieve section {section} after multiple attempts"
            })
        
        # Enhanced prospecto retrieval
        async def get_prospecto_section(section=None):
            urls_to_try = [
                (f"{self.base_url}/docSegmentado/contenido/2", {"nregistro": nregistro, "seccion": section} if section else {"nregistro": nregistro}),
                (f"https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html", None),
                (f"https://cima.aemps.es/cima/dochtml/p/{nregistro}/Prospecto_{nregistro}.html", None),
                (f"https://cima.aemps.es/cima/pdfs/p/{nregistro}/P_{nregistro}.pdf", None)
            ]
            
            for url, params in urls_to_try:
                for attempt in range(2):  # Try each URL up to 2 times
                    try:
                        if params:
                            async with session.get(url, params=params) as response:
                                if response.status == 200:
                                    content = await response.text()
                                    if content and len(content) > 100:
                                        logger.info(f"Successfully retrieved prospecto from {url}")
                                        return {"prospecto_html": content}
                        else:
                            async with session.get(url) as response:
                                if response.status == 200:
                                    content = await response.text()
                                    if content and len(content) > 100:
                                        logger.info(f"Successfully retrieved prospecto from {url}")
                                        return {"prospecto_html": content}
                                        
                        if attempt < 1:
                            await asyncio.sleep(1)
                    except Exception as e:
                        logger.warning(f"Error accessing prospecto at {url} (attempt {attempt+1}): {str(e)}")
                        if attempt < 1:
                            await asyncio.sleep(1)
            
            # If all attempts fail, return informative message
            errors.append("No se pudo obtener el prospecto")
            return {
                "prospecto_html": f"Prospecto disponible en: https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html",
                "error": "Failed to retrieve prospecto after multiple attempts"
            }
        
        # Execute basic info request first - we need this for sure
        basic_info = await get_basic_info()
        details["basic"] = basic_info
        
        # Only fetch sections if we got basic info successfully
        if "error" not in basic_info:
            # Limit concurrent section requests to avoid overwhelming the API
            semaphore = asyncio.Semaphore(3)  # Reduced from 5 to 3 concurrent requests
            
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
            
            # Add direct links to documents for better user experience
            details["document_links"] = {
                "ficha_tecnica": f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html",
                "prospecto": f"https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html"
            }
            
            # Add errors if there were any, for debugging
            if errors:
                details["extraction_errors"] = errors
        
        logger.info(f"Completed fetching details for nregistro: {nregistro}")
        return details

    def format_medication_info(self, index: int, med: Dict, details: Dict, query_intent: Optional[QueryIntent] = None, is_critical: bool = False) -> str:
        """Improved medication information formatting with better section handling"""
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
        
        # Format sections with proper handling of missing data - enhanced for critical references
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
        """Enhanced context retrieval with improved error handling and fallbacks"""
        logger.info(f"Getting context for query: '{query}'")
        cache_key = f"{query}_{n_results}"
        
        # Use cache if available
        if cache_key in self.reference_cache:
            cached_results = self.reference_cache[cache_key]
            
            # Extract query intent from cached data if available
            query_intent = None
            if len(cached_results) > 0 and len(cached_results[0]) > 2:
                query_intent = cached_results[0][2]  # Third element is query intent
                
            context_parts = [self.format_medication_info(i, med, details, query_intent) 
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

        # Enhanced formulation detection (needed for both search methods)
        formulation_info = self.detect_formulation_type(query)
        
        # Create a search instance
        search_implementation = MedicationSearchGraph()
        
        try:
            # Execute the search with retry mechanism
            max_retries = 3
            results = None
            quality = "unknown"
            query_intent = None
            
            for attempt in range(max_retries):
                try:
                    results, quality, query_intent = await search_implementation.execute_search(query)
                    if results:
                        break
                    logger.warning(f"Search attempt {attempt+1} returned no results, retrying...")
                    await asyncio.sleep(1)  # Backoff before retry
                except Exception as e:
                    logger.error(f"Search attempt {attempt+1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                    else:
                        # Last attempt failed, will continue with fallback methods
                        pass
            
            if results:
                logger.info(f"Search found {len(results)} results with quality: {quality}")
                
                # Get/create aiohttp session
                session = await self.get_session()
                
                # Fetch detailed information for each result with semaphore to limit concurrency
                semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent requests to avoid overwhelming the API
                cached_results = []
                
                async def fetch_details_with_retry(med):
                    async with semaphore:
                        nregistro = med.get("nregistro")
                        if not nregistro:
                            return None
                            
                        # Try multiple times with exponential backoff
                        for retry in range(3):
                            try:
                                details = await self.get_medication_details(nregistro)
                                # Validate we got some essential data
                                if details and isinstance(details, dict) and (
                                    "basic" in details or 
                                    "composicion" in details or 
                                    "indicaciones" in details or
                                    "prospecto" in details
                                ):
                                    return (med, details, query_intent)
                                logger.warning(f"Got incomplete details for {nregistro} on attempt {retry+1}, retrying...")
                                await asyncio.sleep(1 * (retry + 1))
                            except Exception as e:
                                logger.error(f"Error fetching details for {nregistro} on attempt {retry+1}: {str(e)}")
                                if retry < 2:  # Don't sleep on the last attempt
                                    await asyncio.sleep(1 * (retry + 1))
                        
                        # If all retries failed, create a minimal details object with direct links
                        return (med, {
                            "basic": med,
                            "extraction_errors": ["Failed to retrieve detailed information after multiple attempts"],
                            "document_links": {
                                "ficha_tecnica": f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html",
                                "prospecto": f"https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html"
                            }
                        }, query_intent)
                
                # Create tasks for each result
                detail_tasks = [fetch_details_with_retry(med) for med in results[:n_results]]
                detail_results = await asyncio.gather(*detail_tasks)
                
                # Filter out None results
                cached_results = [result for result in detail_results if result is not None]
                
                # Store in cache
                if cached_results:
                    self.reference_cache[cache_key] = cached_results
                    
                    # Format content with special handling for important references
                    # For high-quality matches, emphasize more sections
                    context_parts = []
                    for i, (med, details, _) in enumerate(cached_results, 1):
                        # Check if this is a critical reference (high relevance score)
                        is_critical = False
                        if quality == "high" and i == 1:  # First result of high quality match
                            is_critical = True
                            
                        context_part = self.format_medication_info(i, med, details, query_intent, is_critical)
                        context_parts.append(context_part)
                    
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
        except Exception as e:
            logger.error(f"Error in search, falling back to original method: {str(e)}")
        
        # Fallback to original search method
        # Get/create aiohttp session
        session = await self.get_session()
        
        # Check if the query is an exact medication name in uppercase (like "MINOXIDIL BIORGA")
        uppercase_names = formulation_info.get("uppercase_names", [])
        if uppercase_names:
            logger.info(f"Original method: Detected uppercase medication name pattern: {uppercase_names[0]}")
            # Try direct search for the medication first - with retries
            for attempt in range(3):
                try:
                    search_url = f"{self.base_url}/medicamentos"
                    async with session.get(search_url, params={"nombre": uppercase_names[0]}) as response:
                        if response.status == 200:
                            data = await response.json()
                            if isinstance(data, dict) and "resultados" in data and data["resultados"]:
                                # Found direct match
                                med = data["resultados"][0]  # Take first match
                                nregistro = med.get("nregistro")
                                if nregistro:
                                    logger.info(f"Found direct match for {uppercase_names[0]}: {nregistro}")
                                    # Get complete details for this medication
                                    details = await self.get_medication_details(nregistro)
                                    # Add to cache and return formatted info
                                    self.reference_cache[cache_key] = [(med, details)]
                                    return self.format_medication_info(1, med, details, None, True)  # Mark as critical
                    break  # If we get here with no results, no need to retry
                except Exception as e:
                    logger.error(f"Error in direct uppercase medication search (attempt {attempt+1}): {str(e)}")
                    if attempt < 2:  # Don't sleep on the last attempt
                        await asyncio.sleep(1 * (attempt + 1))
        
        # Create a placeholder with links to try manually if no results were found
        if uppercase_names:
            # Create a placeholder with links to try manually
            placeholder = f"""
[Referencia: {uppercase_names[0]} (No encontrado en CIMA)]

No se encontraron resultados exactos para '{uppercase_names[0]}' en CIMA.

Sugerencias:
- Verificar el nombre exacto del medicamento
- Intentar buscar por principio activo: {formulation_info.get("active_principle") if formulation_info.get("active_principle") else "No detectado"}
- Consultar directamente en la web oficial: https://cima.aemps.es/

Nota: Es posible que este medicamento esté registrado con un nombre ligeramente diferente o que sea un medicamento extranjero no incluido en CIMA.
"""
            return placeholder

        # If all else fails, return standard message
        return "No se encontraron medicamentos relevantes en CIMA para esta consulta. Por favor intente con términos más específicos o verifique que el principio activo o medicamento esté registrado en CIMA."

    def detect_formulation_type(self, query: str) -> Dict[str, Any]:
        """Enhanced formulation type detection with improved patterns"""
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

    async def _comprehensive_medication_search(self, session, query, active_principle=None, form_type=None):
        """Enhanced search strategy using multiple approaches with better error handling"""
        all_results = []
        seen_nregistros = set()
        search_url = f"{self.base_url}/medicamentos"
        
        # Extract all potential search terms
        search_terms = self._extract_search_terms(query)
        
        # Helper function for search with error handling
        async def execute_search(params, description):
            results = []
            try:
                logger.info(f"Executing {description} search with params: {params}")
                retry_count = 2  # Allow a retry in case of temporary issues
                
                for attempt in range(retry_count):
                    try:
                        async with session.get(search_url, params=params) as response:
                            if response.status == 200:
                                try:
                                    data = await response.json()
                                    if isinstance(data, dict) and "resultados" in data:
                                        results = data.get("resultados", [])
                                        logger.info(f"{description} search returned {len(results)} results")
                                        return results
                                except Exception as e:
                                    logger.warning(f"Error parsing JSON in {description} search: {str(e)}")
                                    # If JSON parsing fails, try to continue
                            else:
                                logger.warning(f"Non-200 status in {description} search: {response.status}")
                    except Exception as e:
                        logger.warning(f"Error in attempt {attempt+1} for {description} search: {str(e)}")
                        if attempt < retry_count - 1:
                            # Wait before retry
                            await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Final error in {description} search: {str(e)}")
            return results

        # For uppercase name patterns (like "MINOXIDIL BIORGA"), try direct search first
        uppercase_names = re.findall(r'\b[A-Z]{2,}\s+[A-Z]{2,}\b', query.upper())
        if uppercase_names:
            logger.info(f"Searching for uppercase name: {uppercase_names[0]}")
            results = await execute_search({"nombre": uppercase_names[0]}, "uppercase name")
            self._add_unique_results(results, all_results, seen_nregistros)
            
            # If we find a direct match, prioritize this highly
            if all_results:
                logger.info(f"Found direct match for {uppercase_names[0]}")
                return all_results
        
        # 1. Search by exact active principle
        if active_principle:
            results = await execute_search({"principiosActivos": active_principle}, "active principle")
            self._add_unique_results(results, all_results, seen_nregistros)
            
            # If no results, try variations
            if not all_results and len(active_principle) > 4:
                # Try with variations (like without accents or with different cases)
                variations = [
                    active_principle.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u'),
                    active_principle.lower(),
                    active_principle.upper(),
                    active_principle.capitalize()
                ]
                
                for var in variations:
                    if var != active_principle:
                        results = await execute_search({"principiosActivos": var}, f"active principle variation '{var}'")
                        if self._add_unique_results(results, all_results, seen_nregistros) and len(all_results) >= 3:
                            break
                            
            # Try searching by different fields
            if len(all_results) < 3:
                fields = ["practiv1", "nombre"]
                for field in fields:
                    params = {field: active_principle}
                    results = await execute_search(params, f"{field} with active principle")
                    self._add_unique_results(results, all_results, seen_nregistros)
        
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
                
                # If this term gave good results, prioritize it for active principle search
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
        
        # 5. Special cases - For minoxidil or other explicit names
        if not all_results and active_principle and active_principle.lower() == "minoxidil":
            # Try more direct searches for minoxidil
            specific_searches = [
                {"nregistro": "78929"},  # MINOXIDIL BIORGA
                {"nombre": "MINOXIDIL"},
                {"nombre": "MINOXIDIL BIORGA"}
            ]
            
            for params in specific_searches:
                results = await execute_search(params, f"special case search for minoxidil")
                if self._add_unique_results(results, all_results, seen_nregistros):
                    logger.info("Found special case match for minoxidil")
                    break
        
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
                concentration_match = re.search(r'(\d+(?:[,.]\d+)?\s*(?:%|mg|g|ml|mcg|UI|unidades))', query.lower())
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
        """Extract potential search terms from the query with improved pattern matching"""
        # Patterns for potential medication names and active ingredients
        patterns = [
            r'([A-Z][a-záéíóúñ]+(?:\s[a-záéíóúñ]+){0,3})',  # Capitalized words
            r'(\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|UI|unidades))',  # Dosages
            r'([A-Za-záéíóúñ]+\+[A-Za-záéíóúñ]+)',  # Combinations with +
            r'(MINOXIDIL\s+BIORGA)'  # Specific case for MINOXIDIL BIORGA
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

    async def generate_response(self, query: str, context: str, query_intent: Optional[QueryIntent] = None) -> str:
        """Generate response with selected system prompt based on query type and intent"""
        # Extract formulation details for improved prompting
        formulation_info = self.detect_formulation_type(query)
        
        # Select the appropriate system prompt based on query type
        if formulation_info["is_prospecto"]:
            system_prompt = self.prospecto_prompt
        elif query_intent and query_intent.intent_type != "general":
            # Create a focused prompt for specific information queries
            active_principle = formulation_info.get("active_principle", "el medicamento")
            system_prompt = self.focused_information_prompt.replace(
                "{information_type}", query_intent.description
            ).replace(
                "{active_principle}", active_principle
            )
        else:
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

        # If it's a specific information query, add extra instructions
        if query_intent and query_intent.intent_type != "general":
            prompt += f"""
Dado que la consulta es específicamente sobre {query_intent.description}, 
centra tu respuesta en esta información y proporciona todos los detalles relevantes de manera clara y directa.
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
        # Get context with improved query understanding
        search_implementation = MedicationSearchGraph()
        results, quality, query_intent = await search_implementation.execute_search(question)
        
        context = await self.get_relevant_context(question)
        answer = await self.generate_response(question, context, query_intent)
        
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

@dataclass
class CIMAExpertAgent:
    openai_client: AsyncOpenAI
    reference_cache: Dict[str, str] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    base_url: str = Config.CIMA_BASE_URL
    session: aiohttp.ClientSession = None
    max_tokens: int = 14000  # Reserve tokens for prompt and response
    use_langgraph: bool = True  # Use improved search by default
    
    system_prompt = """Eres un experto farmacéutico especializado en medicamentos registrados en CIMA (Centro de Información online de Medicamentos de la AEMPS).

Tu objetivo es proporcionar información precisa y detallada sobre medicamentos en respuesta a las consultas del usuario. Debes:

1. Responder con información basada exclusivamente en los datos oficiales de CIMA
2. Citar las fuentes utilizando el formato: [Ref X: Nombre del medicamento (Nº Registro)]
3. Explicar la información de manera clara y estructurada
4. Proporcionar enlaces a las fichas técnicas y prospectos cuando sea relevante
5. Advertir cuando la información consultada no esté disponible en CIMA

Tipos de consultas que puedes responder:
- Indicaciones y contraindicaciones de medicamentos
- Posología y forma de administración
- Composición y excipientes
- Efectos adversos y precauciones
- Datos de conservación y caducidad
- Comparativas entre medicamentos similares
- Alternativas terapéuticas dentro del mismo grupo

Si se menciona un medicamento concreto, prioriza la información sobre ese medicamento específico. Si la consulta es general, proporciona información sobre los medicamentos más representativos o utilizados para esa indicación.

Para consultas muy específicas sobre formulación magistral, recomienda consultar la pestaña "Formulación Magistral" de esta aplicación.

Responde de manera profesional pero accesible, recordando que tus respuestas pueden ser leídas tanto por profesionales sanitarios como por pacientes.
"""
    
    focused_system_prompt = """Eres un experto farmacéutico especializado en información sobre {information_type} de medicamentos registrados en CIMA.

Has recibido una consulta específica sobre {information_type} de {active_principle}. Responde de manera clara, directa y completa, centrándote exclusivamente en esta consulta.

La información debe estar basada exclusivamente en los datos oficiales de CIMA. Cita las fuentes utilizando el formato: [Ref X: Nombre del medicamento (Nº Registro)].

Estructura tu respuesta de manera lógica:
1. Comienza con un resumen conciso de la información solicitada
2. Proporciona los detalles completos, organizados por puntos o párrafos según sea más apropiado
3. Incluye cualquier advertencia o consideración especial relevante
4. Concluye con recomendaciones generales si son pertinentes

Utiliza un lenguaje preciso pero accesible, recordando que tu respuesta puede ser leída tanto por profesionales sanitarios como por pacientes.
"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.reference_cache = {}
        self.base_url = Config.CIMA_BASE_URL
        self.conversation_history = []
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
            "minoxidil"
        ]
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
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
    
    async def chat(self, query: str) -> Dict[str, str]:
        """Process a chat query with CIMA context"""
        # Reuse the FormulationAgent's context retrieval logic
        formulation_agent = FormulationAgent(self.openai_client)
        formulation_agent.session = await self.get_session()  # Share session for efficiency
        formulation_agent.use_langgraph = self.use_langgraph  # Use same search method
        
        try:
            # Get intent from enhanced search
            search_implementation = MedicationSearchGraph()
            results, quality, query_intent = await search_implementation.execute_search(query)
            
            # Get context using the FormulationAgent's enhanced methods
            context = await formulation_agent.get_relevant_context(query, n_results=3)
            
            # Create chat history context
            history_text = ""
            if self.conversation_history:
                history_text = "HISTORIAL DE CONVERSACIÓN:\n"
                for msg in self.conversation_history[-3:]:  # Only use last 3 messages for context
                    role = "Usuario" if msg["role"] == "user" else "Asistente"
                    history_text += f"{role}: {msg['content']}\n\n"
            
            # Determine if we should use the focused system prompt
            if query_intent and query_intent.intent_type != "general":
                # Get formulation info to extract active principle
                formulation_info = formulation_agent.detect_formulation_type(query)
                active_principle = formulation_info.get("active_principle", "el medicamento")
                
                # Use focused system prompt
                system_prompt = self.focused_system_prompt.replace(
                    "{information_type}", query_intent.description
                ).replace(
                    "{active_principle}", active_principle
                )
            else:
                # Use general system prompt
                system_prompt = self.system_prompt
            
            # Create prompt
            prompt = f"""
Analiza el siguiente contexto y el historial de conversación para responder a la consulta del usuario:

CONTEXTO DE CIMA:
{context}

{history_text}

CONSULTA ACTUAL:
{query}
"""

            # Add intent information if available
            if query_intent and query_intent.intent_type != "general":
                prompt += f"""
TIPO DE INFORMACIÓN SOLICITADA: {query_intent.description}

Dado que el usuario está preguntando específicamente sobre {query_intent.description} de un medicamento,
centra tu respuesta en proporcionar información detallada y completa sobre este aspecto.
"""

            prompt += """
Proporciona información detallada y precisa basada en los datos de CIMA. Cita las fuentes utilizando el formato [Ref X: Nombre del medicamento (Nº Registro)].
Si desconoces la respuesta o no hay información suficiente en el contexto, indícalo claramente.
"""
            
            # Generate response
            chat_completion = await self.openai_client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            answer = chat_completion.choices[0].message.content
            
            # Create direct links for references in response
            pattern = r'\[Ref (\d+): ([^()]+) \(Nº Registro: (\d+)\)\]'
            
            def replace_with_link(match):
                ref_num = match.group(1)
                med_name = match.group(2)
                reg_num = match.group(3)
                return f'[Ref {ref_num}: {med_name} (Nº Registro: {reg_num})](https://cima.aemps.es/cima/dochtml/ft/{reg_num}/FT_{reg_num}.html)'
            
            answer_with_links = re.sub(pattern, replace_with_link, answer)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer_with_links})
            
            # Ensure we don't keep too much history
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return {
                "answer": answer_with_links,
                "context": context
            }
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            raise
        finally:
            # Don't close the shared session here, let the main application manage it
            pass
    
    async def close(self):
        """Close the aiohttp session to free resources with proper error handling"""
        if self.session and not self.session.closed:
            try:
                await self.session.close()
                # Give the event loop time to clean up connections
                await asyncio.sleep(0.25)
            except Exception as e:
                logger.error(f"Error closing session: {str(e)}")
                # Ensure session is marked as closed even if there was an error
                self.session = None
