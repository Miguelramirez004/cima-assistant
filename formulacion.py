from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Union
from openai import AsyncOpenAI
from config import Config
import aiohttp
import re
import json
import asyncio
from datetime import datetime

@dataclass
class FormulationAgent:
    openai_client: AsyncOpenAI
    base_url: str = Config.CIMA_BASE_URL
    reference_cache: Dict[str, List[Dict]] = field(default_factory=dict)
    session: aiohttp.ClientSession = None

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
        """Optimized method to fetch medication details with concurrent requests"""
        session = await self.get_session()
        sections_of_interest = {
            "2": "composicion",
            "4.1": "indicaciones",
            "4.2": "posologia_procedimiento",
            "4.3": "contraindicaciones",
            "4.4": "advertencias",
            "4.5": "interacciones",
            "4.6": "embarazo_lactancia",
            "4.8": "efectos_adversos",
            "5.1": "propiedades_farmacodinamicas",
            "5.2": "propiedades_farmacocineticas",
            "6.1": "excipientes",
            "6.3": "conservacion",
            "6.4": "especificaciones",
            "6.5": "envase",
            "6.6": "eliminacion"
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
                print(f"Error retrieving basic details: {str(e)}")
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
                print(f"Error retrieving section {section}: {str(e)}")
            return (key, {"contenido": f"No disponible"})
        
        async def get_images():
            image_url = f"{self.base_url}/medicamento/fotos"
            try:
                async with session.get(image_url, params={"nregistro": nregistro}) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, dict):
                            return result
                        elif isinstance(result, list):
                            return {"fotos": result}
            except Exception as e:
                print(f"Error retrieving images: {str(e)}")
            return {"error": "Unable to retrieve images"}
        
        # Execute basic info request first - we need this for sure
        basic_info = await get_basic_info()
        details["basic"] = basic_info
        
        # Only fetch sections if we got basic info successfully
        if "error" not in basic_info:
            # Limit concurrent section requests to avoid overwhelming the API
            semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
            
            async def limited_section_fetch(section, key):
                async with semaphore:
                    return await get_section(section, key)
            
            # Create tasks for each section (with concurrency limit)
            section_tasks = [limited_section_fetch(section, key) for section, key in sections_of_interest.items()]
            
            # Execute section tasks concurrently and process results
            section_results = await asyncio.gather(*section_tasks)
            for key, value in section_results:
                details[key] = value
            
            # Get images as a separate task
            details["imagenes"] = await get_images()
        
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
        
        # Process query
        query_lower = query.lower()
        
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
            "active_principle": active_principle
        }

    def format_medication_info(self, index: int, med: Dict, details: Dict) -> str:
        """
        Enhanced medication information formatting with more detailed structure
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
        def get_section_content(section_key):
            section_data = details.get(section_key, {})
            if not isinstance(section_data, dict):
                return "No disponible"
            
            # Limit content length to reduce context size
            content = section_data.get('contenido', 'No disponible')
            if len(content) > 500:  # Truncate very long sections
                return content[:497] + "..."
            return content
        
        # Construct a more compact formatted reference to reduce context size
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

URL FICHA TÉCNICA:
https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html
"""
        # Removed several less critical sections to reduce context size
        return reference

    async def get_relevant_context(self, query: str, n_results: int = 3) -> str:
        """
        Optimized context retrieval with more efficient search and error handling
        - Reduced n_results default from 5 to 3 to improve performance
        - Added concurrent search processing
        """
        cache_key = f"{query}_{n_results}"
        
        # Enhanced formulation detection
        formulation_info = self.detect_formulation_type(query)
        formulation_type = formulation_info["form_type"]
        active_principle = formulation_info["active_principle"]
        
        if cache_key in self.reference_cache:
            cached_results = self.reference_cache[cache_key]
            context = [self.format_medication_info(i, med, details) 
                      for i, (med, details) in enumerate(cached_results, 1)]
            return "\n".join(context)

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
                            return data.get("resultados", [])
            except Exception as e:
                print(f"Error in search with params {params}: {str(e)}")
            return []
        
        # Define all search strategies
        search_params = [
            {"nombre": active_principle},
            {"principiosActivos": active_principle},
            {"formaFarmaceutica": formulation_type}
        ]
        
        # Execute searches concurrently
        search_results = await asyncio.gather(*(search_medications(params) for params in search_params))
        
        # Combine results while removing duplicates
        all_results = []
        seen_nregistros = set()
        
        for results in search_results:
            for med in results:
                if isinstance(med, dict) and med.get("nregistro"):
                    nregistro = med.get("nregistro")
                    if nregistro not in seen_nregistros:
                        seen_nregistros.add(nregistro)
                        all_results.append(med)
        
        # Fallback if no results found
        if not all_results:
            # Try a more generic search
            first_word = query.split()[0] if query.split() else ""
            if len(first_word) > 3:
                fallback_results = await search_medications({"nombre": first_word})
                for med in fallback_results:
                    if isinstance(med, dict) and med.get("nregistro"):
                        nregistro = med.get("nregistro")
                        if nregistro not in seen_nregistros:
                            seen_nregistros.add(nregistro)
                            all_results.append(med)
        
        # Limit results
        results = all_results[:n_results]
        cached_results = []

        if results:
            # Fetch details for each medication concurrently
            semaphore = asyncio.Semaphore(3)  # Limit concurrent API calls to 3
            
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
            
            # Format context, with reduced context size
            context = [self.format_medication_info(i, med, details) 
                      for i, (med, details) in enumerate(cached_results, 1)]
            
            return "\n".join(context)
        
        return "No se encontraron resultados relevantes."

    async def generate_response(self, query: str, context: str) -> str:
        """
        Enhanced response generation with performance optimizations
        """
        # Extract formulation details for improved prompting
        formulation_info = self.detect_formulation_type(query)
        
        # Streamlined prompt
        prompt = f"""
Analiza el siguiente contexto para generar una formulación magistral detallada:

CONTEXTO:
{context}

DETALLES DE LA FORMULACIÓN SOLICITADA:
- Tipo de formulación: {formulation_info["form_type"]}
- Vía de administración: {formulation_info["admin_route"]}
- Principio(s) activo(s): {formulation_info["active_principle"]}
- Concentración solicitada: {formulation_info["concentration"] if formulation_info["concentration"] else "No especificada"}

CONSULTA ORIGINAL:
{query}

Por favor, genera una formulación magistral completa siguiendo la estructura indicada en las instrucciones del sistema. Cita las fuentes CIMA usando el formato [Ref X: Nombre del medicamento (Nº Registro)].
"""

        # Use a lower temperature for faster response
        response = await self.openai_client.chat.completions.create(
            model=Config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5  # Lower temperature for faster processing
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
        Optimized implementation for obtaining medication information using multiple endpoints
        """
        cache_key = f"query_{query}"
        if cache_key in self.reference_cache:
            return self.reference_cache[cache_key]

        # Extract search terms more efficiently
        potential_terms = self._extract_search_terms(query)
        
        # Get the session
        session = await self.get_session()
        
        # Perform searches concurrently for better performance
        tasks = []
        
        # 1. Direct name search
        tasks.append(self._search_medications_by_name(session, query))
        
        # 2. Search by extracted terms - limit to first 2 terms for performance
        for term in potential_terms[:2]:
            if term.lower() != query.lower():  # Avoid duplicate searches
                tasks.append(self._search_medications_by_name(session, term))
        
        # Execute all searches concurrently
        results = await asyncio.gather(*tasks)
        
        # Filter out empty results and combine
        all_results = [r for r in results if r]
        
        # Add ficha técnica search only if needed
        if len(all_results) < 2:
            ft_results = await self._search_in_ficha_tecnica(session, query)
            if ft_results:
                all_results.append(ft_results)
        
        # Combine results
        combined_results = "\n\n".join(all_results)
        
        # Truncate if too large to improve response time
        if len(combined_results) > 10000:
            combined_results = combined_results[:10000] + "\n\n[Resultados truncados por longitud]"
        
        self.reference_cache[cache_key] = combined_results
        return combined_results

    def _extract_search_terms(self, query: str) -> List[str]:
        """
        Extrae términos de búsqueda potenciales del query del usuario - optimizado
        """
        # Patrones para nombres de medicamentos y principios activos
        patterns = [
            r'([A-Z][a-záéíóúñ]+(?:\s[a-záéíóúñ]+){0,3})',  # Palabras capitalizadas
            r'(\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|UI|unidades))',  # Dosificaciones
            r'([A-Za-záéíóúñ]+\+[A-Za-záéíóúñ]+)'  # Combinaciones con +
        ]
        
        # Lista de palabras clave comunes que pueden indicar medicamentos
        keywords = [
            "medicamento", "fármaco", "principio activo", "comprimido", "pastilla", 
            "cápsula", "jarabe", "solución", "inyectable", "pomada", "crema", "gel"
        ]
        
        # Extraer usando patrones
        potential_terms = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            potential_terms.extend([m.strip() for m in matches if len(m.strip()) > 3])
        
        # Extraer palabras potencialmente relevantes de forma más eficiente
        words = query.split()
        for i in range(len(words)):
            # Palabras individuales que son largas y no son comunes
            if len(words[i]) > 4 and words[i].lower() not in keywords:
                potential_terms.append(words[i])
            
            # Bi-gramas (pares de palabras) - solo los primeros para mejorar rendimiento
            if i < len(words) - 1 and i < 3 and len(words[i]) > 3 and len(words[i+1]) > 3:
                potential_terms.append(f"{words[i]} {words[i+1]}")
        
        # Eliminar duplicados y retornar lista de términos potenciales (limitada a 5)
        seen = set()
        return [x for x in potential_terms if x.lower() not in seen and not seen.add(x.lower())][:5]

    async def _search_medications_by_name(self, session, query: str) -> str:
        """
        Busca medicamentos por nombre - versión optimizada
        """
        search_url = f"{self.base_url}/medicamentos"
        results = []
        
        # Semaphore to limit concurrent detail fetches
        detail_semaphore = asyncio.Semaphore(3)
        
        # Create task for search by name
        async def search_by_params(params):
            try:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, dict) and "resultados" in data:
                            return data.get("resultados", [])[:3]  # Limit to top 3 results
            except Exception as e:
                print(f"Error en búsqueda con parámetros {params}: {str(e)}")
            return []
        
        # Execute search by name
        meds = await search_by_params({"nombre": query})
        
        # If no results, try search by active ingredient
        if not meds:
            meds = await search_by_params({"practiv1": query})
        
        # If still no results, try partial search
        if not meds and len(query) > 4:
            partial_query = query[:4]
            meds = await search_by_params({"nombre": partial_query})
        
        # Limit to top 3 medications for better performance
        meds = meds[:3]
        
        # Function to fetch medication details with semaphore
        async def fetch_med_details(med):
            async with detail_semaphore:
                if isinstance(med, dict) and med.get("nregistro"):
                    return await self._get_reduced_medication_details(session, med)
            return None
        
        # Fetch details concurrently
        if meds:
            detail_tasks = [fetch_med_details(med) for med in meds]
            detail_results = await asyncio.gather(*detail_tasks)
            results = [r for r in detail_results if r]
        
        return "\n\n".join(results) if results else ""

    async def _search_in_ficha_tecnica(self, session, query: str) -> str:
        """
        Busca en fichas técnicas - versión optimizada
        """
        results = []
        search_url = f"{self.base_url}/buscarEnFichaTecnica"
        
        # Limitar secciones y palabras para mejor rendimiento
        sections = ["4.1", "4.2"]  # Reducir secciones a las más importantes
        words = [word for word in query.split() if len(word) > 3][:2]  # Limitar a primeras 2 palabras relevantes
        
        search_body = []
        for section in sections:
            for word in words:
                search_body.append({
                    "seccion": section,
                    "texto": word,
                    "contiene": 1
                })
        
        if not search_body:
            return ""
        
        try:
            async with session.post(search_url, json=search_body) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, dict) and "resultados" in data:
                        meds = data.get("resultados", [])[:2]  # Limitar a 2 resultados
                        
                        # Semaphore for detail fetches
                        detail_semaphore = asyncio.Semaphore(2)
                        
                        async def fetch_med_details(med):
                            async with detail_semaphore:
                                if isinstance(med, dict) and med.get("nregistro"):
                                    return await self._get_reduced_medication_details(session, med)
                            return None
                        
                        detail_tasks = [fetch_med_details(med) for med in meds]
                        detail_results = await asyncio.gather(*detail_tasks)
                        results = [r for r in detail_results if r]
        except Exception as e:
            print(f"Error en búsqueda en ficha técnica: {str(e)}")
        
        return "\n\n".join(results) if results else ""

    async def _get_reduced_medication_details(self, session, med: Dict) -> str:
        """
        Obtiene un conjunto reducido de detalles clave del medicamento para mejor rendimiento
        """
        if not isinstance(med, dict):
            return ""
        
        nregistro = med.get("nregistro")
        if not nregistro:
            return ""
        
        # 1. Información básica del medicamento
        basic_info = {}
        detail_url = f"{self.base_url}/medicamento"
        
        try:
            async with session.get(detail_url, params={"nregistro": nregistro}) as response:
                if response.status == 200:
                    result = await response.json()
                    if isinstance(result, dict):
                        basic_info = result
        except Exception as e:
            print(f"Error obteniendo información básica para {nregistro}: {str(e)}")
            return ""
        
        # 2. Información de secciones clave (reducidas para mejor rendimiento)
        key_sections = {
            "4.1": "indicaciones",
            "4.2": "posologia",
            "4.3": "contraindicaciones",
            "4.8": "efectos_adversos"
        }
        
        sections_data = {}
        
        # Fetch sections concurrently
        async def fetch_section(section, key):
            try:
                section_url = f"{self.base_url}/docSegmentado/contenido/1"
                async with session.get(section_url, params={"nregistro": nregistro, "seccion": section}) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, dict) and "contenido" in result:
                            content = result.get("contenido", "")
                            # Truncate long content
                            if len(content) > 500:
                                content = content[:497] + "..."
                            return (key, content)
            except Exception as e:
                print(f"Error obteniendo sección {section} para {nregistro}: {str(e)}")
            return (key, "No disponible")
        
        section_tasks = [fetch_section(section, key) for section, key in key_sections.items()]
        section_results = await asyncio.gather(*section_tasks)
        
        for key, content in section_results:
            sections_data[key] = content
        
        # Formatear la información de forma reducida
        name = med.get("nombre", basic_info.get("nombre", "No disponible"))
        pactivos = med.get("pactivos", basic_info.get("pactivos", "No disponible"))
        lab = basic_info.get("labtitular", "No disponible")
        
        # URL de la ficha técnica
        ficha_url = f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html"
        
        # Construir un bloque de información más compacto
        info_block = [
            f"[Ref: {name} (Nº Registro: {nregistro})]",
            f"Nombre: {name}",
            f"Principios activos: {pactivos}",
            f"Laboratorio: {lab}"
        ]
        
        # Añadir secciones clave
        if sections_data.get("indicaciones"):
            info_block.append(f"INDICACIONES:\n{sections_data['indicaciones']}")
        
        if sections_data.get("posologia"):
            info_block.append(f"POSOLOGÍA:\n{sections_data['posologia']}")
        
        if sections_data.get("contraindicaciones"):
            info_block.append(f"CONTRAINDICACIONES:\n{sections_data['contraindicaciones']}")
        
        if sections_data.get("efectos_adversos"):
            info_block.append(f"EFECTOS ADVERSOS:\n{sections_data['efectos_adversos']}")
        
        # Añadir enlace a ficha técnica
        info_block.append(f"Ficha técnica completa: {ficha_url}")
        
        return "\n\n".join(info_block)

    def _format_date(self, unix_timestamp):
        """
        Formatea fechas de Unix timestamp a formato legible
        """
        if not unix_timestamp:
            return "No disponible"
        
        try:
            # Las fechas en la API CIMA están en Unix timestamp (millisegundos)
            dt = datetime.fromtimestamp(unix_timestamp / 1000)
            return dt.strftime("%d/%m/%Y")
        except:
            return str(unix_timestamp)

    async def chat(self, message: str) -> Dict[str, str]:
        """
        Procesamiento de chat optimizado
        """
        try:
            # Obtener información relevante de CIMA
            context = await self.get_medication_info(message)
            
            # Si no hay resultados, intentar con términos más genéricos pero limitados
            if not context:
                generic_terms = self._extract_generic_terms(message)
                for term in generic_terms[:2]:  # Limitar a 2 términos para mejorar rendimiento
                    term_context = await self.get_medication_info(term)
                    if term_context:
                        context = term_context
                        break
            
            # Streamline prompt to reduce size
            prompt = f"""
Consulta: {message}

Contexto relevante de CIMA:
{context if context else "No se encontró información específica en CIMA para esta consulta."}

Responde de manera concisa, citando las fuentes específicas del contexto cuando sea relevante.
"""
            
            # Usar solo las últimas 4 mensajes para mantener el contexto más ligero
            recent_history = self.conversation_history[-4:] if len(self.conversation_history) > 4 else self.conversation_history
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                *recent_history,
                {"role": "user", "content": prompt}
            ]

            # Generar respuesta con temperatura reducida para mejor rendimiento
            response = await self.openai_client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=messages,
                temperature=0.5
            )

            assistant_response = response.choices[0].message.content
            
            # Añadir enlaces directos a CIMA
            pattern = r'\[Ref: ([^()]+) \(Nº Registro: (\d+)\)\]'
            
            def replace_with_link(match):
                med_name = match.group(1)
                reg_num = match.group(2)
                return f'[Ref: {med_name} (Nº Registro: {reg_num})](https://cima.aemps.es/cima/dochtml/ft/{reg_num}/FT_{reg_num}.html)'
            
            assistant_response_with_links = re.sub(pattern, replace_with_link, assistant_response)
            
            # Actualizar historial de conversación
            self.conversation_history.extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": assistant_response}
            ])

            return {
                "answer": assistant_response_with_links,
                "context": context[:1000] + "..." if len(context) > 1000 else context,  # Truncate context for UI display
                "history": self.conversation_history
            }
        except Exception as e:
            error_message = f"Lo siento, ha ocurrido un error al procesar tu consulta: {str(e)}"
            print(f"Error en chat: {str(e)}")
            return {
                "answer": error_message,
                "context": f"Error: {str(e)}",
                "history": self.conversation_history
            }

    def _extract_generic_terms(self, message: str) -> List[str]:
        """
        Extrae términos genéricos más eficiente
        """
        # Categorías comunes de medicamentos
        categories = [
            "analgésico", "antibiótico", "antiinflamatorio", "antidepresivo", 
            "ansiolítico", "antihistamínico", "antihipertensivo"
        ]
        
        # Palabras clave de condiciones médicas comunes
        conditions = [
            "hipertensión", "diabetes", "asma", "ansiedad", "depresión",
            "dolor", "infección", "alergia"
        ]
        
        # Extracción más eficiente
        message_lower = message.lower()
        
        # 1. Buscar categorías y condiciones
        generic_terms = [term for term in categories + conditions if term in message_lower]
        
        # 2. Añadir palabras largas (posibles nombres de medicamentos)
        words = [word for word in message.split() if len(word) > 4 and word.lower() not in generic_terms]
        generic_terms.extend(words[:3])  # Limitar a 3 palabras para rendimiento
        
        return generic_terms

    def clear_history(self):
        """
        Limpiar historial de conversación
        """
        self.conversation_history = []
        
    async def close(self):
        """Close the aiohttp session to free resources"""
        if self.session and not self.session.closed:
            await self.session.close()