from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Union
from openai import AsyncOpenAI
from config import Config
import aiohttp
import ssl
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
        formulation_types = {
            "suspension": ["suspension", "suspensión", "suspens"],
            "solucion": ["solucion", "solución", "sol."],
            "papelillos": ["papelillos", "sobres", "polvos"],
            "pomada": ["pomada", "unguento", "crema", "pasta"],
            "gel": ["gel", "hidrogel"],
            "supositorios": ["supositorio", "rectal"],
            "colirio": ["colirio", "oftálmico", "oftalmico", "gotas oculares"],
            "jarabe": ["jarabe", "formula pediátrica", "formula pediatrica"],
            "cápsulas": ["cápsulas", "capsulas", "encapsulado"],
            "emulsion": ["emulsion", "emulsión", "locion", "loción"]
        }
        
        # Dictionary for pharmaceutical paths
        admin_routes = {
            "oral": ["oral", "vía oral", "via oral", "por boca"],
            "topica": ["tópica", "topica", "cutánea", "cutanea"],
            "oftalmico": ["oftálmico", "oftalmico", "ocular"],
            "rectal": ["rectal", "vía rectal", "via rectal"],
            "nasal": ["nasal", "intranasal"],
            "otico": ["ótico", "otico", "auricular"],
            "vaginal": ["vaginal", "intravaginal"],
            "parenteral": ["parenteral", "inyectable", "inyección", "inyeccion"]
        }
        
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