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

    async def get_medication_details(self, nregistro: str, session) -> Dict:
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
        
        # Get basic medication information
        detail_url = f"{self.base_url}/medicamento"
        try:
            async with session.get(detail_url, params={"nregistro": nregistro}) as response:
                if response.status == 200:
                    result = await response.json()
                    # Check if result is a dictionary before using get()
                    if isinstance(result, dict):
                        details["basic"] = result
                    else:
                        details["basic"] = {"error": "Unexpected response format"}
        except Exception as e:
            print(f"Error retrieving basic details: {str(e)}")
            details["basic"] = {"error": str(e)}
            
        # Get technical information for each section
        for section, key in sections_of_interest.items():
            tech_url = f"{self.base_url}/docSegmentado/contenido/1"
            params = {"nregistro": nregistro, "seccion": section}
            try:
                async with session.get(tech_url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, dict):
                            details[key] = result
                        else:
                            details[key] = {"contenido": "Formato inesperado"}
            except Exception as e:
                print(f"Error retrieving section {section}: {str(e)}")
                details[key] = {"contenido": f"Error: {str(e)}"}
                    
        # Get additional information like images if available
        try:
            image_url = f"{self.base_url}/medicamento/fotos"
            async with session.get(image_url, params={"nregistro": nregistro}) as response:
                if response.status == 200:
                    result = await response.json()
                    if isinstance(result, dict):
                        details["imagenes"] = result
                    elif isinstance(result, list):
                        details["imagenes"] = {"fotos": result}
                    else:
                        details["imagenes"] = {"error": "Formato inesperado"}
        except Exception as e:
            print(f"Error retrieving images: {str(e)}")
            
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
            return section_data.get('contenido', 'No disponible')
        
        # Construct the formatted reference
        reference = f"""
[Referencia {index}: {med_name} (Nº Registro: {nregistro})]

INFORMACIÓN BÁSICA:
- Nombre: {med_name}
- Número de registro: {nregistro}
- Laboratorio titular: {lab_titular}
- Fecha de autorización: {fecha_autorizacion}
- Principios activos: {med.get('pactivos', 'No disponible')}

COMPOSICIÓN:
{get_section_content('composicion')}

EXCIPIENTES:
{get_section_content('excipientes')}

INDICACIONES TERAPÉUTICAS:
{get_section_content('indicaciones')}

POSOLOGÍA Y ADMINISTRACIÓN:
{get_section_content('posologia_procedimiento')}

CONTRAINDICACIONES:
{get_section_content('contraindicaciones')}

ADVERTENCIAS Y PRECAUCIONES:
{get_section_content('advertencias')}

INTERACCIONES:
{get_section_content('interacciones')}

EMBARAZO Y LACTANCIA:
{get_section_content('embarazo_lactancia')}

EFECTOS ADVERSOS:
{get_section_content('efectos_adversos')}

PROPIEDADES FARMACODINÁMICAS:
{get_section_content('propiedades_farmacodinamicas')}

PROPIEDADES FARMACOCINÉTICAS:
{get_section_content('propiedades_farmacocineticas')}

CONSERVACIÓN:
{get_section_content('conservacion')}

ESPECIFICACIONES:
{get_section_content('especificaciones')}

ENVASE:
{get_section_content('envase')}

URL FICHA TÉCNICA:
https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html
"""
        return reference

    async def get_relevant_context(self, query: str, n_results: int = 5) -> str:
        """
        Enhanced context retrieval with better search parameters and error handling
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

        # Use TCPConnector without SSL verification
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            search_url = f"{self.base_url}/medicamentos"
            
            # Enhanced search parameters
            search_params = {
                "nombre": active_principle,
                "formaFarmaceutica": formulation_type,
                "principiosActivos": active_principle
            }
            
            # Add parallel searches for better retrieval
            all_results = []
            
            # Search by name with better error handling
            try:
                async with session.get(search_url, params={"nombre": active_principle}) as response:
                    if response.status == 200:
                        meds = await response.json()
                        # Check if meds is a dictionary before using get()
                        if isinstance(meds, dict) and "resultados" in meds:
                            result_list = meds.get("resultados", [])
                            if isinstance(result_list, list):
                                all_results.extend(result_list)
                        else:
                            print("Warning: Unexpected response format from name search")
            except Exception as e:
                print(f"Error in name search: {str(e)}")
            
            # Search by active principle
            try:
                async with session.get(search_url, params={"principiosActivos": active_principle}) as response:
                    if response.status == 200:
                        meds = await response.json()
                        if isinstance(meds, dict) and "resultados" in meds:
                            result_list = meds.get("resultados", [])
                            if isinstance(result_list, list):
                                new_results = [med for med in result_list 
                                             if not any(med.get("nregistro") == r.get("nregistro") for r in all_results)]
                                all_results.extend(new_results)
                        else:
                            print("Warning: Unexpected response format from active principle search")
            except Exception as e:
                print(f"Error in active principle search: {str(e)}")
            
            # Search by pharmaceutical form
            try:
                async with session.get(search_url, params={"formaFarmaceutica": formulation_type}) as response:
                    if response.status == 200:
                        meds = await response.json()
                        if isinstance(meds, dict) and "resultados" in meds:
                            result_list = meds.get("resultados", [])
                            if isinstance(result_list, list):
                                new_results = [med for med in result_list 
                                             if not any(med.get("nregistro") == r.get("nregistro") for r in all_results)]
                                all_results.extend(new_results)
                        else:
                            print("Warning: Unexpected response format from form search")
            except Exception as e:
                print(f"Error in pharmaceutical form search: {str(e)}")
            
            # Fallback if no results found
            if not all_results:
                # Try a more generic search
                try:
                    first_word = query.split()[0] if query.split() else ""
                    if len(first_word) > 3:
                        async with session.get(search_url, params={"nombre": first_word}) as response:
                            if response.status == 200:
                                meds = await response.json()
                                if isinstance(meds, dict) and "resultados" in meds:
                                    result_list = meds.get("resultados", [])
                                    if isinstance(result_list, list):
                                        all_results.extend(result_list[:n_results])
                except Exception as e:
                    print(f"Error in fallback search: {str(e)}")
            
            # Limit results and get details
            results = all_results[:n_results]
            cached_results = []

            # Process results with semaphore to limit concurrent API calls
            semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent requests
            
            async def get_med_details(med):
                async with semaphore:
                    if not isinstance(med, dict) or not med.get("nregistro"):
                        return None
                    details = await self.get_medication_details(med["nregistro"], session)
                    return (med, details)
            
            # Get details for all medications concurrently
            tasks = [get_med_details(med) for med in results]
            completed_tasks = await asyncio.gather(*tasks)
            
            # Filter out None results and add to cache
            cached_results = [result for result in completed_tasks if result is not None]
            self.reference_cache[cache_key] = cached_results
            
            # Format context
            context = [self.format_medication_info(i, med, details) 
                      for i, (med, details) in enumerate(cached_results, 1)]

        return "\n".join(context) if context else "No se encontraron resultados relevantes."

    async def generate_response(self, query: str, context: str) -> str:
        """
        Enhanced response generation with better prompting
        """
        # Extract formulation details for improved prompting
        formulation_info = self.detect_formulation_type(query)
        
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

Por favor, genera una formulación magistral completa siguiendo la estructura indicada en las instrucciones del sistema. Cita las fuentes CIMA utilizando el formato [Ref X: Nombre del medicamento (Nº Registro)].
"""

        response = await self.openai_client.chat.completions.create(
            model=Config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": self.system_prompt},
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

@dataclass
class CIMAExpertAgent:
    openai_client: AsyncOpenAI
    reference_cache: Dict[str, str] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    base_url: str = Config.CIMA_BASE_URL
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.reference_cache = {}
        self.base_url = Config.CIMA_BASE_URL
        self.conversation_history = []

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

    async def get_medication_info(self, query: str) -> str:
        """
        Implementación mejorada para obtener información de medicamentos usando múltiples endpoints
        """
        cache_key = f"query_{query}"
        if cache_key in self.reference_cache:
            return self.reference_cache[cache_key]

        # Extraer nombres de medicamentos y principios activos potenciales
        potential_terms = self._extract_search_terms(query)
        
        # Resultado combinado de todas las búsquedas
        all_results = []
        
        # Use safe connector for SSL
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            # 1. Búsqueda por nombre directa
            direct_results = await self._search_medications_by_name(session, query)
            if direct_results:
                all_results.append(direct_results)
            
            # 2. Búsqueda por términos extraídos
            for term in potential_terms:
                if term.lower() != query.lower():  # Evitar búsquedas duplicadas
                    term_results = await self._search_medications_by_name(session, term)
                    if term_results:
                        all_results.append(term_results)
            
            # 3. Búsqueda avanzada en fichas técnicas si no hay suficientes resultados
            if len(all_results) < 2:
                ft_results = await self._search_in_ficha_tecnica(session, query)
                if ft_results:
                    all_results.append(ft_results)
            
            # Combinar resultados
            combined_results = "\n\n".join(all_results)
            self.reference_cache[cache_key] = combined_results
            
            return combined_results

    def _extract_search_terms(self, query: str) -> List[str]:
        """
        Extrae términos de búsqueda potenciales del query del usuario
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
            "cápsula", "jarabe", "solución", "inyectable", "pomada", "crema", "gel",
            "parche", "supositorio", "aerosol", "inhalador", "suspensión"
        ]
        
        # Extraer usando patrones
        potential_terms = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            potential_terms.extend([m.strip() for m in matches if len(m.strip()) > 3])
        
        # Extraer palabras potencialmente relevantes
        words = query.split()
        for i in range(len(words)):
            # Palabras individuales que son largas y no son comunes
            if len(words[i]) > 4 and words[i].lower() not in keywords:
                potential_terms.append(words[i])
            
            # Bi-gramas (pares de palabras)
            if i < len(words) - 1 and len(words[i]) > 3 and len(words[i+1]) > 3:
                potential_terms.append(f"{words[i]} {words[i+1]}")
        
        # Eliminar duplicados y retornar lista de términos potenciales
        return list(set(potential_terms))

    async def _search_medications_by_name(self, session, query: str) -> str:
        """
        Busca medicamentos por nombre usando el endpoint de medicamentos
        """
        search_url = f"{self.base_url}/medicamentos"
        results = []
        
        # Parámetros de búsqueda por nombre
        try:
            async with session.get(search_url, params={"nombre": query}) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, dict) and "resultados" in data:
                        meds = data.get("resultados", [])
                        for med in meds[:5]:  # Limitamos a 5 resultados
                            if isinstance(med, dict) and med.get("nregistro"):
                                details = await self._get_complete_medication_details(session, med)
                                if details:
                                    results.append(details)
        except Exception as e:
            print(f"Error en búsqueda por nombre '{query}': {str(e)}")
        
        # Si no hay resultados, intentar búsqueda por principio activo
        if not results:
            try:
                async with session.get(search_url, params={"practiv1": query}) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, dict) and "resultados" in data:
                            meds = data.get("resultados", [])
                            for med in meds[:5]:
                                if isinstance(med, dict) and med.get("nregistro"):
                                    details = await self._get_complete_medication_details(session, med)
                                    if details:
                                        results.append(details)
            except Exception as e:
                print(f"Error en búsqueda por principio activo '{query}': {str(e)}")
        
        # Si todavía no hay resultados, intentar búsqueda parcial
        if not results and len(query) > 4:
            try:
                partial_query = query[:4]
                async with session.get(search_url, params={"nombre": partial_query}) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, dict) and "resultados" in data:
                            meds = data.get("resultados", [])
                            for med in meds[:3]:
                                if isinstance(med, dict) and med.get("nregistro"):
                                    details = await self._get_complete_medication_details(session, med)
                                    if details:
                                        results.append(details)
            except Exception as e:
                print(f"Error en búsqueda parcial '{partial_query}': {str(e)}")
        
        return "\n\n".join(results) if results else ""

    async def _search_in_ficha_tecnica(self, session, query: str) -> str:
        """
        Busca en fichas técnicas usando el endpoint buscarEnFichaTecnica
        """
        results = []
        search_url = f"{self.base_url}/buscarEnFichaTecnica"
        
        # Preparar búsqueda en varias secciones relevantes de la ficha técnica
        sections = ["4.1", "4.2", "4.3", "4.4", "5.1"]
        search_body = []
        
        # Palabras de la consulta para buscar
        words = [word for word in query.split() if len(word) > 3]
        
        # Crear búsquedas para cada sección y palabra relevante
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
                        meds = data.get("resultados", [])
                        for med in meds[:5]:
                            if isinstance(med, dict) and med.get("nregistro"):
                                details = await self._get_complete_medication_details(session, med)
                                if details:
                                    results.append(details)
        except Exception as e:
            print(f"Error en búsqueda en ficha técnica: {str(e)}")
        
        return "\n\n".join(results) if results else ""

    async def _get_complete_medication_details(self, session, med: Dict) -> str:
        """
        Obtiene detalles completos de un medicamento usando varios endpoints
        """
        if not isinstance(med, dict):
            return ""
        
        nregistro = med.get("nregistro")
        if not nregistro:
            return ""
        
        # 1. Información básica del medicamento
        basic_info = await self._get_basic_medication_info(session, nregistro)
        if not basic_info:
            return ""
        
        # 2. Información de secciones de la ficha técnica
        sections_info = await self._get_technical_sections(session, nregistro)
        
        # 3. Información de presentaciones
        presentations_info = await self._get_presentations(session, nregistro)
        
        # Formatear toda la información
        return self._format_medication_complete_info(med, basic_info, sections_info, presentations_info)

    async def _get_basic_medication_info(self, session, nregistro: str) -> Dict:
        """
        Obtiene información básica del medicamento
        """
        basic_info = {}
        detail_url = f"{self.base_url}/medicamento"
        
        try:
            async with session.get(detail_url, params={"nregistro": nregistro}) as response:
                if response.status == 200:
                    result = await response.json()
                    if isinstance(result, dict):
                        return result
        except Exception as e:
            print(f"Error obteniendo información básica para {nregistro}: {str(e)}")
        
        return basic_info

    async def _get_technical_sections(self, session, nregistro: str) -> Dict:
        """
        Obtiene secciones de la ficha técnica
        """
        sections_of_interest = {
            "1": "composicion",
            "2": "forma_farmaceutica",
            "3": "datos_clinicos",
            "4.1": "indicaciones",
            "4.2": "posologia",
            "4.3": "contraindicaciones",
            "4.4": "advertencias",
            "4.5": "interacciones",
            "4.6": "embarazo_lactancia",
            "4.8": "efectos_adversos",
            "5.1": "propiedades_farmacodinamicas",
            "5.2": "propiedades_farmacocineticas",
            "6.1": "excipientes",
            "6.3": "conservacion",
            "6.4": "especificaciones"
        }
        
        sections_data = {}
        
        for section, key in sections_of_interest.items():
            try:
                section_url = f"{self.base_url}/docSegmentado/contenido/1"
                async with session.get(section_url, params={"nregistro": nregistro, "seccion": section}) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, dict) and "contenido" in result:
                            sections_data[key] = result.get("contenido", "")
            except Exception as e:
                print(f"Error obteniendo sección {section} para {nregistro}: {str(e)}")
        
        return sections_data

    async def _get_presentations(self, session, nregistro: str) -> List[Dict]:
        """
        Obtiene presentaciones del medicamento
        """
        presentations = []
        presentation_url = f"{self.base_url}/presentaciones"
        
        try:
            async with session.get(presentation_url, params={"nregistro": nregistro}) as response:
                if response.status == 200:
                    result = await response.json()
                    if isinstance(result, dict) and "resultados" in result:
                        return result.get("resultados", [])
        except Exception as e:
            print(f"Error obteniendo presentaciones para {nregistro}: {str(e)}")
        
        return presentations

    def _format_medication_complete_info(self, med: Dict, basic_info: Dict, sections_info: Dict, presentations_info: List[Dict]) -> str:
        """
        Formatea toda la información del medicamento
        """
        if not isinstance(med, dict) or not isinstance(basic_info, dict):
            return ""
        
        # Información básica
        nregistro = med.get("nregistro", "")
        name = med.get("nombre", basic_info.get("nombre", "No disponible"))
        pactivos = med.get("pactivos", basic_info.get("pactivos", "No disponible"))
        lab = basic_info.get("labtitular", "No disponible")
        
        # URL de la ficha técnica
        ficha_url = f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html"
        
        # Construir el bloque de información
        info_block = [
            f"[Ref: {name} (Nº Registro: {nregistro})]",
            f"Nombre: {name}",
            f"Principios activos: {pactivos}",
            f"Laboratorio: {lab}"
        ]
        
        # Añadir estado de autorización
        estado = basic_info.get("estado", {})
        if isinstance(estado, dict):
            if "aut" in estado:
                info_block.append(f"Fecha autorización: {self._format_date(estado.get('aut'))}")
            if "susp" in estado:
                info_block.append(f"Fecha suspensión: {self._format_date(estado.get('susp'))}")
            if "rev" in estado:
                info_block.append(f"Fecha revocación: {self._format_date(estado.get('rev'))}")
        
        # Añadir información de secciones
        section_titles = {
            "composicion": "COMPOSICIÓN",
            "forma_farmaceutica": "FORMA FARMACÉUTICA",
            "datos_clinicos": "DATOS CLÍNICOS",
            "indicaciones": "INDICACIONES TERAPÉUTICAS",
            "posologia": "POSOLOGÍA Y FORMA DE ADMINISTRACIÓN",
            "contraindicaciones": "CONTRAINDICACIONES",
            "advertencias": "ADVERTENCIAS Y PRECAUCIONES",
            "interacciones": "INTERACCIONES",
            "embarazo_lactancia": "EMBARAZO Y LACTANCIA",
            "efectos_adversos": "EFECTOS ADVERSOS",
            "propiedades_farmacodinamicas": "PROPIEDADES FARMACODINÁMICAS",
            "propiedades_farmacocineticas": "PROPIEDADES FARMACOCINÉTICAS",
            "excipientes": "EXCIPIENTES",
            "conservacion": "CONSERVACIÓN",
            "especificaciones": "ESPECIFICACIONES"
        }
        
        for key, title in section_titles.items():
            content = sections_info.get(key, "")
            if content and len(content.strip()) > 10:
                # Limitar longitud de cada sección para evitar respuestas excesivamente largas
                if len(content) > 800:
                    content = content[:797] + "..."
                info_block.append(f"{title}:\n{content}")
        
        # Añadir presentaciones
        if presentations_info:
            info_block.append("PRESENTACIONES:")
            for i, pres in enumerate(presentations_info[:3], 1):
                cn = pres.get("cn", "")
                nombre_pres = pres.get("nombre", "")
                estado_pres = "Comercializada" if pres.get("comerc") else "No comercializada"
                info_block.append(f"{i}. CN: {cn} - {nombre_pres} - {estado_pres}")
        
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
        Procesa la consulta del usuario y genera una respuesta basada en CIMA
        """
        try:
            # Obtener información relevante de CIMA
            context = await self.get_medication_info(message)
            
            # Si no hay resultados, intentar con términos más genéricos
            if not context:
                generic_terms = self._extract_generic_terms(message)
                for term in generic_terms:
                    term_context = await self.get_medication_info(term)
                    if term_context:
                        context = term_context
                        break
            
            # Preparar el prompt para el modelo
            prompt = f"""
Para responder a esta consulta, usa la información proporcionada en el contexto CIMA.

Consulta: {message}

Contexto relevante de CIMA:
{context if context else "No se encontró información específica en CIMA para esta consulta."}

Responde de manera precisa, citando las fuentes específicas del contexto con el formato [Ref: Nombre del medicamento (Nº Registro)].
Incluye enlaces directos a las fichas técnicas cuando sea relevante.
No inventes información que no aparezca en el contexto proporcionado.
Si no hay información suficiente, indica qué tipo de información adicional sería útil para responder mejor.
"""
            
            # Usar solo las últimas 10 mensajes para mantener el contexto manejable
            recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                *recent_history,
                {"role": "user", "content": prompt}
            ]

            # Generar respuesta
            response = await self.openai_client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=messages,
                temperature=0.7
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
                "context": context,
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
        Extrae términos genéricos que podrían ser útiles para una búsqueda más amplia
        """
        # Categorías comunes de medicamentos
        categories = [
            "analgésico", "antibiótico", "antiinflamatorio", "antidepresivo", 
            "ansiolítico", "antihistamínico", "antihipertensivo", "hipnótico",
            "antiácido", "anticoagulante", "antidiabético", "antipsicótico",
            "corticoide", "diurético", "laxante", "mucolítico"
        ]
        
        # Palabras clave de condiciones médicas comunes
        conditions = [
            "hipertensión", "diabetes", "asma", "ansiedad", "depresión",
            "dolor", "infección", "alergia", "insomnio", "artritis",
            "migraña", "úlcera", "reflujo", "colesterol", "epilepsia"
        ]
        
        # Extraer términos que coincidan con categorías o condiciones
        generic_terms = []
        message_lower = message.lower()
        
        for term in categories + conditions:
            if term in message_lower:
                generic_terms.append(term)
        
        # Extraer otros términos potencialmente útiles
        words = message.split()
        for word in words:
            if len(word) > 4 and word.lower() not in generic_terms:
                generic_terms.append(word)
        
        return generic_terms

    def clear_history(self):
        """
        Limpiar historial de conversación
        """
        self.conversation_history = []