"""
Improved CIMA API client implementation following the official AEMPS documentation.
This module provides cleaner, more reliable API interactions with proper parameter handling.
"""

import logging
import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional, Union
from config import Config

logger = logging.getLogger(__name__)

class CIMAClient:
    """
    Client for interacting with the CIMA API following official documentation.
    """
    
    def __init__(self, base_url: str = Config.CIMA_BASE_URL):
        self.base_url = base_url
        self.session = None
        
        # Section mapping for document segments (remove dots as per API requirements)
        self.section_map = {
            "2": "2",        # composicion
            "4.1": "41",     # indicaciones
            "4.2": "42",     # posologia
            "4.3": "43",     # contraindicaciones
            "4.4": "44",     # advertencias
            "4.5": "45",     # interacciones
            "4.6": "46",     # embarazo_lactancia
            "4.8": "48",     # efectos_adversos
            "5.1": "51",     # propiedades_farmacodinamicas
            "5.2": "52",     # propiedades_farmacocineticas
            "5.3": "53",     # datos_preclinicos
            "6.1": "61",     # excipientes
            "6.2": "62",     # incompatibilidades
            "6.3": "63",     # conservacion
            "6.4": "64",     # especificaciones
            "6.5": "65",     # envase
            "6.6": "66",     # eliminacion
            "7": "7",        # titular_autorizacion
            "8": "8",        # numero_autorizacion
            "9": "9",        # fecha_autorizacion
            "10": "10"       # fecha_revision
        }
        
        # Known medications with registration numbers
        self.known_medications = {
            "minoxidil biorga": "78929",
            "minoxidil": "78929",
            "biorga": "78929",
            "regaine": "81897",
            "ibuprofeno": "43513",
            "paracetamol": "64033"
        }
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session with optimal settings"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                ssl=True,
                limit=5,
                keepalive_timeout=30,
                force_close=False
            )
            timeout = aiohttp.ClientTimeout(
                total=60,
                connect=20,
                sock_connect=20,
                sock_read=30
            )
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                raise_for_status=False
            )
        return self.session
    
    async def close(self):
        """Close the client session"""
        if self.session and not self.session.closed:
            try:
                await self.session.close()
                await asyncio.sleep(0.25)  # Allow cleanup
            except Exception as e:
                logger.error(f"Error closing session: {str(e)}")
                self.session = None
    
    async def search_medications(self, 
                                params: Dict[str, Any], 
                                description: str = "search") -> List[Dict[str, Any]]:
        """
        Search medications with proper API parameters

        Args:
            params: Search parameters
            description: Description for logging

        Returns:
            List of medication dictionaries
        """
        session = await self.get_session()
        results = []
        search_url = f"{self.base_url}/medicamentos"
        
        # Ensure required parameters are present
        if "pagina" not in params:
            params["pagina"] = 1
        
        # Standard headers
        headers = {"Accept": "application/json"}
        
        try:
            logger.info(f"Executing {description} with params: {params}")
            
            # Add rate limiting awareness
            await asyncio.sleep(0.2)  # Prevent overloading the API
            
            async with session.get(search_url, params=params, headers=headers) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        if isinstance(data, dict) and "resultados" in data:
                            results = data.get("resultados", [])
                            logger.info(f"{description} returned {len(results)} results")
                    except Exception as e:
                        logger.warning(f"Error parsing JSON in {description}: {str(e)}")
                        
                        # If JSON parsing fails, try to log the response text for debugging
                        try:
                            text = await response.text()
                            logger.warning(f"Response text (first 200 chars): {text[:200]}")
                        except:
                            pass
                else:
                    logger.warning(f"Non-200 status in {description}: {response.status}")
        except Exception as e:
            logger.error(f"Error in {description}: {str(e)}")
        
        return results
    
    async def get_medication_details(self, nregistro: str) -> Dict[str, Any]:
        """
        Get medication details with its registration number

        Args:
            nregistro: Registration number

        Returns:
            Dictionary with medication details
        """
        session = await self.get_session()
        details = {}
        
        # Standard headers
        headers = {"Accept": "application/json"}
        
        try:
            logger.info(f"Fetching details for nregistro: {nregistro}")
            detail_url = f"{self.base_url}/medicamento"
            
            # Add rate limiting awareness
            await asyncio.sleep(0.2)
            
            async with session.get(detail_url, params={"nregistro": nregistro}, headers=headers) as response:
                if response.status == 200:
                    try:
                        basic_info = await response.json()
                        if isinstance(basic_info, dict):
                            details["basic"] = basic_info
                            logger.info(f"Successfully retrieved basic info for {nregistro}")
                        else:
                            logger.warning(f"Unexpected response format for basic info: {type(basic_info)}")
                    except Exception as e:
                        logger.warning(f"Error parsing basic info JSON: {str(e)}")
                        # Try to log the response text for debugging
                        try:
                            text = await response.text()
                            logger.warning(f"Response text (first 200 chars): {text[:200]}")
                        except:
                            pass
                else:
                    logger.warning(f"Non-200 status for basic info: {response.status}")
        except Exception as e:
            logger.error(f"Error retrieving basic details: {str(e)}")
            details["basic"] = {"nregistro": nregistro, "error": "Unable to retrieve basic details"}
        
        # Fetch technical document sections
        details.update(await self.get_technical_sections(nregistro))
        
        # Fetch prospecto
        details["prospecto"] = await self.get_prospecto(nregistro)
        
        # Add direct links for convenience
        details["document_links"] = {
            "ficha_tecnica": f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FT_{nregistro}.html",
            "prospecto": f"https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html"
        }
        
        return details
    
    async def get_technical_sections(self, nregistro: str) -> Dict[str, Any]:
        """
        Get technical document sections for a medication

        Args:
            nregistro: Registration number

        Returns:
            Dictionary of section data
        """
        session = await self.get_session()
        sections = {}
        
        # Standard headers
        headers = {"Accept": "application/json"}
        
        # Sections to fetch
        section_keys = [
            "composicion", "indicaciones", "posologia_procedimiento", "contraindicaciones",
            "advertencias", "interacciones", "embarazo_lactancia", "efectos_adversos",
            "excipientes", "conservacion"
        ]
        
        # Mapping from section keys to actual section IDs
        section_id_map = {
            "composicion": "2",
            "indicaciones": "4.1",
            "posologia_procedimiento": "4.2",
            "contraindicaciones": "4.3",
            "advertencias": "4.4",
            "interacciones": "4.5",
            "embarazo_lactancia": "4.6",
            "efectos_adversos": "4.8",
            "excipientes": "6.1",
            "conservacion": "6.3"
        }
        
        # Limit concurrent section requests
        semaphore = asyncio.Semaphore(3)
        
        async def fetch_section(section_key: str):
            """Fetch a single section with rate limiting"""
            async with semaphore:
                section_id = section_id_map.get(section_key)
                if not section_id:
                    return section_key, {"contenido": "No disponible", "error": "Invalid section key"}
                
                api_section_id = self.section_map.get(section_id, section_id.replace(".", ""))
                tech_url = f"{self.base_url}/docSegmentado/contenido/1"
                params = {"nregistro": nregistro, "seccion": api_section_id}
                
                # Add rate limiting
                await asyncio.sleep(0.2)
                
                try:
                    async with session.get(tech_url, params=params, headers=headers) as response:
                        if response.status == 200:
                            try:
                                result = await response.json()
                                if isinstance(result, dict) and "contenido" in result:
                                    return section_key, result
                                else:
                                    logger.warning(f"Unexpected section response format for {section_key}")
                                    return section_key, {"contenido": "No disponible", "error": "Unexpected format"}
                            except Exception as e:
                                logger.warning(f"Error parsing section {section_key}: {str(e)}")
                                return section_key, {"contenido": "No disponible", "error": f"Parse error: {str(e)}"}
                        else:
                            logger.warning(f"Non-200 status for section {section_key}: {response.status}")
                            return section_key, {"contenido": "No disponible", "error": f"Status: {response.status}"}
                except Exception as e:
                    logger.error(f"Error fetching section {section_key}: {str(e)}")
                    return section_key, {"contenido": "No disponible", "error": str(e)}
        
        # Create tasks for all sections
        tasks = [fetch_section(key) for key in section_keys]
        results = await asyncio.gather(*tasks)
        
        # Process results
        for key, value in results:
            sections[key] = value
        
        return sections
    
    async def get_prospecto(self, nregistro: str) -> Dict[str, Any]:
        """
        Get medication prospecto

        Args:
            nregistro: Registration number

        Returns:
            Dictionary with prospecto data
        """
        session = await self.get_session()
        
        # Standard headers
        headers = {"Accept": "application/json"}
        
        # Try to get prospecto through API first
        try:
            logger.info(f"Fetching prospecto for nregistro: {nregistro}")
            prospecto_url = f"{self.base_url}/docSegmentado/contenido/2"
            
            # Add rate limiting
            await asyncio.sleep(0.2)
            
            async with session.get(prospecto_url, params={"nregistro": nregistro}, headers=headers) as response:
                if response.status == 200:
                    try:
                        prospecto_data = await response.json()
                        if isinstance(prospecto_data, dict) and "contenido" in prospecto_data:
                            return {"prospecto_html": prospecto_data["contenido"]}
                        else:
                            logger.warning(f"Unexpected prospecto format: {type(prospecto_data)}")
                    except Exception as e:
                        logger.warning(f"Error parsing prospecto JSON: {str(e)}")
                else:
                    logger.warning(f"Non-200 status for prospecto: {response.status}")
        except Exception as e:
            logger.error(f"Error retrieving prospecto: {str(e)}")
        
        # Fallback: Direct link
        return {
            "prospecto_html": f"Prospecto disponible en: https://cima.aemps.es/cima/dochtml/p/{nregistro}/P_{nregistro}.html",
            "error": "Failed to retrieve prospecto through API"
        }
    
    async def direct_lookup(self, medication_name: str) -> List[Dict[str, Any]]:
        """
        Perform direct lookup for known medications

        Args:
            medication_name: Name of the medication to look up

        Returns:
            List of medication results
        """
        if medication_name.lower() in self.known_medications:
            nregistro = self.known_medications[medication_name.lower()]
            logger.info(f"Direct lookup for known medication: {medication_name} -> {nregistro}")
            
            session = await self.get_session()
            headers = {"Accept": "application/json"}
            
            try:
                medication_url = f"{self.base_url}/medicamento"
                params = {"nregistro": nregistro}
                
                async with session.get(medication_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        try:
                            med_data = await response.json()
                            if isinstance(med_data, dict) and "nregistro" in med_data:
                                # Add high relevance score for direct lookups
                                med_data["relevance_score"] = 150
                                return [med_data]
                        except Exception as e:
                            logger.warning(f"Error parsing direct lookup: {str(e)}")
            except Exception as e:
                logger.error(f"Error in direct lookup: {str(e)}")
        
        return []
    
    async def search_by_active_principle(self, active_principle: str) -> List[Dict[str, Any]]:
        """
        Search medications by active principle

        Args:
            active_principle: Active ingredient to search for

        Returns:
            List of medication results
        """
        # Prepare parameters according to API documentation
        params = {
            "principiosActivos": active_principle,
            "pagina": 1,
            "tamano": 25
        }
        
        return await self.search_medications(params, "active principle search")
    
    async def search_by_name(self, name: str) -> List[Dict[str, Any]]:
        """
        Search medications by name

        Args:
            name: Medication name to search for

        Returns:
            List of medication results
        """
        # Prepare parameters according to API documentation
        params = {
            "nombre": name,
            "pagina": 1,
            "tamano": 25
        }
        
        return await self.search_medications(params, "name search")
    
    # Special case handling for medications not in CIMA
    def create_custom_result(self, custom_id: str, name: str, active_principle: str) -> Dict[str, Any]:
        """
        Create a custom result for medications not found in CIMA

        Args:
            custom_id: Custom identifier
            name: Medication name
            active_principle: Active principle

        Returns:
            Dictionary with medication data
        """
        return {
            "nregistro": custom_id,
            "nombre": name,
            "pactivos": active_principle,
            "labtitular": "No disponible (fuera de CIMA)",
            "comerc": True,
            "custom": True,
            "relevance_score": 100
        }
    
    def get_melatonina_custom_data(self) -> Dict[str, Any]:
        """Get custom data for melatonina (not in CIMA)"""
        basic_info = {
            "nregistro": "custom_melatonina",
            "nombre": "Melatonina (Suplemento Dietético)",
            "pactivos": "Melatonina",
            "labtitular": "Varios fabricantes (suplemento)",
            "comerc": True,
            "custom": True
        }
        
        sections = {
            "composicion": {"contenido": "Principio activo: Melatonina. Excipientes: Pueden variar según el fabricante."},
            "indicaciones": {"contenido": "Como suplemento alimenticio, la melatonina se utiliza para ayudar a regular el ciclo del sueño. No es un medicamento registrado en CIMA."},
            "posologia_procedimiento": {"contenido": "Generalmente entre 1mg y 5mg al día, tomado 30-60 minutos antes de acostarse. Consultar las indicaciones específicas del fabricante."},
            "contraindicaciones": {"contenido": "Embarazo, lactancia, enfermedades autoinmunes, epilepsia, uso de anticoagulantes. Consultar con un médico antes de su uso."},
            "advertencias": {"contenido": "No es un medicamento. No debe utilizarse como sustituto de un tratamiento médico. Puede causar somnolencia."},
            "interacciones": {"contenido": "Puede interaccionar con anticoagulantes, inmunosupresores, anticonceptivos hormonales y medicamentos que afectan al sistema nervioso central."},
            "embarazo_lactancia": {"contenido": "No recomendado durante el embarazo o la lactancia."},
            "efectos_adversos": {"contenido": "Somnolencia, dolor de cabeza, mareos, náuseas, irritabilidad."},
            "excipientes": {"contenido": "Varía según el fabricante. Consultar el etiquetado del producto específico."},
            "conservacion": {"contenido": "Conservar en lugar fresco y seco, protegido de la luz. Seguir las indicaciones del fabricante."}
        }
        
        prospecto = {
            "prospecto_html": "La melatonina se comercializa como suplemento alimenticio, no como medicamento registrado en CIMA. Consulte el prospecto específico del producto que adquiera."
        }
        
        document_links = {
            "ficha_tecnica": "No disponible - No es un medicamento registrado en CIMA",
            "prospecto": "No disponible - No es un medicamento registrado en CIMA"
        }
        
        return {
            "basic": basic_info,
            **sections,
            "prospecto": prospecto,
            "document_links": document_links,
            "custom_info": "Este producto no es un medicamento registrado en CIMA. Se trata de un suplemento alimenticio."
        }
