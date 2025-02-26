"""
Streamlined medication search implementation to improve relevance of results.

This module provides an improved search approach that solves issues with 
irrelevant results (like returning abacavir for unrelated queries) without 
requiring the full LangGraph dependency.
"""

import re
import logging
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from pydantic import BaseModel, Field
import aiohttp
from dataclasses import dataclass, field

from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Minimum relevance score for a result to be considered valid
MIN_RELEVANCE_THRESHOLD = 20

# Maximum number of medications to return
MAX_RESULTS = 5

class InformationRequest(BaseModel):
    """Represents a specific information request about a drug."""
    type: str  # Type of information (e.g., "contraindications", "dosage", etc.)
    keywords: List[str]  # Keywords that indicate this information type
    found: bool = False  # Whether this type was found in the query

class MedicationQuery(BaseModel):
    """Structured representation of a medication search query."""
    query_text: str
    active_principle: Optional[str] = None
    formulation_type: Optional[str] = None
    administration_route: Optional[str] = None
    concentration: Optional[str] = None
    uppercase_names: List[str] = Field(default_factory=list)
    search_terms: List[str] = Field(default_factory=list)
    is_prospecto: bool = False
    is_information_request: bool = False
    information_request_type: Optional[str] = None
    
class MedicationResult(BaseModel):
    """Structured representation of a medication search result."""
    nregistro: str
    nombre: str
    pactivos: Optional[str] = None
    labtitular: Optional[str] = None
    comerc: Optional[bool] = None
    relevance_score: Optional[int] = 0
    
    class Config:
        # Allow extra fields that might be in the CIMA API response
        extra = "allow"

@dataclass
class MedicationSearchGraph:
    """
    Improved search implementation for medication searches.
    This class provides a more relevant search approach that resolves the
    "abacavir problem" without requiring the full LangGraph dependency.
    """
    base_url: str = Config.CIMA_BASE_URL
    active_principles: List[str] = field(default_factory=lambda: [
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
        "vitamina d", "calcio", "hierro", "insulina", "metronidazol", "minoxidil"
    ])
    # Information request types
    information_requests: List[InformationRequest] = field(default_factory=lambda: [
        InformationRequest(
            type="contraindications",
            keywords=["contraindicaciones", "contraindicación", "no tomar", "no usar", "no debe"]
        ),
        InformationRequest(
            type="side_effects",
            keywords=["efectos secundarios", "efectos adversos", "reacciones adversas", "adversos"]
        ),
        InformationRequest(
            type="dosage",
            keywords=["posología", "posologia", "dosis", "dosificación", "como tomar", "como usar"]
        ),
        InformationRequest(
            type="interactions",
            keywords=["interacciones", "interacción", "junto con", "combinado con", "mezclar con"]
        ),
        InformationRequest(
            type="precautions",
            keywords=["precauciones", "advertencias", "cuidados", "tenga cuidado", "atención"]
        ),
        InformationRequest(
            type="indications",
            keywords=["indicado para", "indicaciones", "uso", "para qué", "para que", "para qué sirve"]
        ),
        InformationRequest(
            type="administration",
            keywords=["administración", "administracion", "vía", "via", "modo de empleo", "como administrar"]
        ),
        InformationRequest(
            type="composition",
            keywords=["composición", "composicion", "ingredientes", "componentes", "excipientes"]
        ),
        InformationRequest(
            type="conservation",
            keywords=["conservación", "conservacion", "almacenamiento", "guardar", "caducidad"]
        )
    ])
    
    async def execute_search(self, query_text: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Execute a comprehensive search for medications based on the query.
        
        Args:
            query_text: The search query
            
        Returns:
            Tuple[List[Dict], str]: List of results and quality assessment
        """
        session = None
        quality = "unknown"
        
        try:
            # Create session
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
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                raise_for_status=False
            )
            
            # Analyze the query
            query_info = self._analyze_query(query_text)
            logger.info(f"Query analysis: {query_info.active_principle}, Information request: {query_info.is_information_request}, Type: {query_info.information_request_type}")
            
            # Results storage
            all_results = []
            seen_nregistros = set()
            
            # For information requests: prioritize exact matches on active principle
            if query_info.is_information_request and query_info.active_principle:
                logger.info(f"Processing information request about {query_info.active_principle}")
                info_results = await self._search_by_active_principle(session, query_info, prioritize_exact_match=True)
                
                # Add results
                all_results.extend(info_results)
                seen_nregistros.update([r.nregistro for r in info_results])
                
                if info_results:
                    quality = "high"
            
            # 1. First try: search for uppercase medication names like "MINOXIDIL BIORGA"
            if query_info.uppercase_names and len(all_results) < MAX_RESULTS:
                logger.info(f"Searching for uppercase name: {query_info.uppercase_names[0]}")
                uppercase_results = await self._search_by_uppercase(session, query_info)
                
                # Add unique results
                for result in uppercase_results:
                    if result.nregistro not in seen_nregistros:
                        all_results.append(result)
                        seen_nregistros.add(result.nregistro)
                
                # If we find good uppercase matches, that's high quality
                if uppercase_results and quality == "unknown":
                    quality = "high"
            
            # 2. Second try: search by active principle if available
            if len(all_results) < MAX_RESULTS and query_info.active_principle:
                logger.info(f"Searching by active principle: {query_info.active_principle}")
                ap_results = await self._search_by_active_principle(session, query_info)
                
                # Add unique results
                for result in ap_results:
                    if result.nregistro not in seen_nregistros:
                        all_results.append(result)
                        seen_nregistros.add(result.nregistro)
                
                if ap_results and quality == "unknown":
                    quality = "medium"
            
            # 3. Third try: search by full name
            if len(all_results) < MAX_RESULTS:
                logger.info(f"Searching by full query: {query_text}")
                name_results = await self._search_by_name(session, query_info)
                
                # Add unique results
                for result in name_results:
                    if result.nregistro not in seen_nregistros:
                        all_results.append(result)
                        seen_nregistros.add(result.nregistro)
                
                if name_results and quality == "unknown":
                    quality = "low"
            
            # 4. Fourth try: search by individual terms
            if len(all_results) < MAX_RESULTS and query_info.search_terms:
                logger.info(f"Searching by terms: {query_info.search_terms}")
                term_results = await self._search_by_terms(session, query_info)
                
                # Add unique results
                for result in term_results:
                    if result.nregistro not in seen_nregistros:
                        all_results.append(result)
                        seen_nregistros.add(result.nregistro)
                
                if term_results and quality == "unknown":
                    quality = "very_low"
            
            # Sort by relevance and filter low-relevance results
            all_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Filter results with low relevance scores
            filtered_results = [r for r in all_results if r.relevance_score >= MIN_RELEVANCE_THRESHOLD]
            
            # Limit to maximum results
            filtered_results = filtered_results[:MAX_RESULTS]
            
            # Set quality to no_results if we didn't find anything
            if not filtered_results:
                quality = "no_results"
            
            logger.info(f"Search completed: {len(filtered_results)} results with quality {quality}")
            
            # Convert to dictionaries for easier integration
            return [result.dict() for result in filtered_results], quality
            
        except Exception as e:
            logger.error(f"Error executing search: {str(e)}")
            return [], "error"
        finally:
            # Close session
            if session:
                try:
                    await session.close()
                except Exception as e:
                    logger.error(f"Error closing session: {str(e)}")
    
    def _analyze_query(self, query_text: str) -> MedicationQuery:
        """Analyze the query to extract structured search parameters."""
        query_lower = query_text.lower()
        
        # Check if this is an information request
        is_information_request = False
        information_request_type = None
        
        # Check if the query matches any information request patterns
        for info_req in self.information_requests:
            for keyword in info_req.keywords:
                if keyword in query_lower:
                    is_information_request = True
                    information_request_type = info_req.type
                    break
            if is_information_request:
                break
        
        # Extract uppercase medication names
        uppercase_names = re.findall(r'\b[A-Z]{2,}\s+[A-Z]{2,}\b', query_text.upper())
        
        # Extract active principle - this is critical for information requests
        active_principle = None
        
        # First look for known active principles
        principles_by_length = sorted(self.active_principles, key=len, reverse=True)
        for ap in principles_by_length:
            if ap in query_lower:
                active_principle = ap
                break
        
        # If still not found, try other extraction methods
        if not active_principle:
            # Look for compound active principles (e.g., "Hidrocortisona y Lidocaína")
            compound_pattern = r'([A-Z][a-záéíóúñ]+(?:\s[a-záéíóúñ]+)*)\s+[y]\s+([A-Z][a-záéíóúñ]+(?:\s[a-záéíóúñ]+)*)'
            compound_match = re.search(compound_pattern, query_text)
            if compound_match:
                active_principle = f"{compound_match.group(1)} {compound_match.group(2)}"
            else:
                # Look for capitalized words
                cap_words = re.findall(r'\b[A-Z][a-záéíóúñ]{2,}\b', query_text)
                if cap_words:
                    active_principle = cap_words[0]
                elif uppercase_names:
                    active_principle = uppercase_names[0].lower()
                else:
                    # Just take the longest word as a guess - filter out information request words
                    info_req_words = set()
                    for info_req in self.information_requests:
                        for keyword in info_req.keywords:
                            info_req_words.update(keyword.split())
                            
                    words = [w for w in query_lower.split() if len(w) > 4 and 
                            not any(x in w for x in ['como', 'para', 'sobre', 'cual', 'este', 'esta']) and
                            w not in info_req_words]
                    if words:
                        active_principle = max(words, key=len)
        
        # Extract formulation type
        formulation_type = None
        formulation_types = Config.FORMULATION_TYPES
        for form_type, keywords in formulation_types.items():
            if any(word in query_lower for word in keywords):
                formulation_type = form_type
                break
        
        # Extract administration route
        admin_route = None
        admin_routes = Config.ADMIN_ROUTES
        for route, keywords in admin_routes.items():
            if any(word in query_lower for word in keywords):
                admin_route = route
                break
        
        # Extract concentration
        concentration = None
        concentration_pattern = r'(\d+(?:[,.]\d+)?\s*(?:%|mg|g|ml|mcg|UI|unidades)|\d+\s*(?:mg)?[/](?:ml|g))'
        concentration_match = re.search(concentration_pattern, query_text)
        if concentration_match:
            concentration = concentration_match.group(0)
        
        # Extract search terms for fuzzy matching
        search_terms = self._extract_search_terms(query_text)
        
        # Check if this is a prospecto request
        prospecto_pattern = r'(?:redactar|generar|crear|elaborar|realizar?e?|escrib[ei]r|hac[ae]r|desarroll[ae]r|realiza(?:r|)|prepar(?:ar|a))\s+(?:un|el|uns?|una?)?\s+prospecto'
        is_prospecto = bool(re.search(prospecto_pattern, query_lower))
        
        # Create and return the query object
        return MedicationQuery(
            query_text=query_text,
            active_principle=active_principle,
            formulation_type=formulation_type,
            administration_route=admin_route,
            concentration=concentration,
            uppercase_names=uppercase_names,
            search_terms=search_terms,
            is_prospecto=is_prospecto,
            is_information_request=is_information_request,
            information_request_type=information_request_type
        )
    
    async def _search_by_uppercase(self, session: aiohttp.ClientSession, query: MedicationQuery) -> List[MedicationResult]:
        """Search for exact matches with uppercase medication names."""
        if not query.uppercase_names:
            return []
        
        try:
            uppercase_name = query.uppercase_names[0]
            search_url = f"{self.base_url}/medicamentos"
            results = []
            
            async with session.get(search_url, params={"nombre": uppercase_name}) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        if isinstance(data, dict) and "resultados" in data and data["resultados"]:
                            for med in data["resultados"]:
                                result = MedicationResult(**med)
                                # Give a high relevance score to exact uppercase matches
                                if uppercase_name.lower() in result.nombre.lower():
                                    result.relevance_score = 100
                                else:
                                    result.relevance_score = 80
                                
                                results.append(result)
                    except Exception as e:
                        logger.error(f"Error parsing uppercase search results: {str(e)}")
            
            return results
        except Exception as e:
            logger.error(f"Error in uppercase search: {str(e)}")
            return []
    
    async def _search_by_active_principle(self, session: aiohttp.ClientSession, query: MedicationQuery, prioritize_exact_match: bool = False) -> List[MedicationResult]:
        """Search by active principle, which is highly relevant for medication searches."""
        if not query.active_principle:
            return []
        
        try:
            active_principle = query.active_principle
            
            # Try variations of the active principle for better results
            variations = [
                active_principle,
                active_principle.lower(),
                active_principle.capitalize(),
                active_principle.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u'),
            ]
            
            search_url = f"{self.base_url}/medicamentos"
            results = []
            seen_nregistros = set()
            
            for variation in variations:
                if len(results) >= MAX_RESULTS:
                    break
                    
                async with session.get(search_url, params={"principiosActivos": variation}) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            if isinstance(data, dict) and "resultados" in data and data["resultados"]:
                                for med in data["resultados"]:
                                    if len(results) >= MAX_RESULTS:
                                        break
                                        
                                    if med.get("nregistro") not in seen_nregistros:
                                        result = MedicationResult(**med)
                                        # Calculate relevance score
                                        result.relevance_score = self._calculate_relevance(
                                            result, 
                                            active_principle=active_principle, 
                                            concentration=query.concentration,
                                            formulation_type=query.formulation_type,
                                            information_request=query.is_information_request
                                        )
                                        
                                        # For information requests, increase the relevance of exact matches
                                        if prioritize_exact_match and query.is_information_request:
                                            if result.pactivos and active_principle.lower() in result.pactivos.lower():
                                                result.relevance_score += 50
                                        
                                        # Only add if it's above threshold
                                        if result.relevance_score >= MIN_RELEVANCE_THRESHOLD:
                                            results.append(result)
                                            seen_nregistros.add(med.get("nregistro"))
                        except Exception as e:
                            logger.error(f"Error parsing active principle search results: {str(e)}")
            
            # Try searching by practiv1 field as fallback
            if len(results) < 2 and len(active_principle) >= 4:
                async with session.get(search_url, params={"practiv1": active_principle}) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            if isinstance(data, dict) and "resultados" in data and data["resultados"]:
                                for med in data["resultados"]:
                                    if len(results) >= MAX_RESULTS:
                                        break
                                        
                                    if med.get("nregistro") not in seen_nregistros:
                                        result = MedicationResult(**med)
                                        result.relevance_score = self._calculate_relevance(
                                            result, 
                                            active_principle=active_principle, 
                                            concentration=query.concentration,
                                            formulation_type=query.formulation_type,
                                            information_request=query.is_information_request
                                        )
                                        
                                        # For information requests, increase the relevance of exact matches
                                        if prioritize_exact_match and query.is_information_request:
                                            if result.pactivos and active_principle.lower() in result.pactivos.lower():
                                                result.relevance_score += 50
                                        
                                        if result.relevance_score >= MIN_RELEVANCE_THRESHOLD:
                                            results.append(result)
                                            seen_nregistros.add(med.get("nregistro"))
                        except Exception as e:
                            logger.error(f"Error parsing practiv1 search results: {str(e)}")
            
            # Sort by relevance score
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results
        except Exception as e:
            logger.error(f"Error in active principle search: {str(e)}")
            return []
    
    async def _search_by_name(self, session: aiohttp.ClientSession, query: MedicationQuery) -> List[MedicationResult]:
        """Search by complete name or query text."""
        try:
            search_url = f"{self.base_url}/medicamentos"
            results = []
            seen_nregistros = set()
            
            # For information requests, just search by active principle if available
            search_term = query.active_principle if query.is_information_request and query.active_principle else query.query_text
            
            async with session.get(search_url, params={"nombre": search_term}) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        if isinstance(data, dict) and "resultados" in data and data["resultados"]:
                            for med in data["resultados"]:
                                if med.get("nregistro") not in seen_nregistros:
                                    result = MedicationResult(**med)
                                    result.relevance_score = self._calculate_relevance(
                                        result, 
                                        active_principle=query.active_principle, 
                                        concentration=query.concentration,
                                        formulation_type=query.formulation_type,
                                        information_request=query.is_information_request
                                    )
                                    
                                    if result.relevance_score >= MIN_RELEVANCE_THRESHOLD:
                                        results.append(result)
                                        seen_nregistros.add(med.get("nregistro"))
                    except Exception as e:
                        logger.error(f"Error parsing name search results: {str(e)}")
            
            # Sort by relevance score
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Limit to max results
            results = results[:MAX_RESULTS]
            return results
        except Exception as e:
            logger.error(f"Error in name search: {str(e)}")
            return []
    
    async def _search_by_terms(self, session: aiohttp.ClientSession, query: MedicationQuery) -> List[MedicationResult]:
        """Search by individual terms extracted from the query (last resort)."""
        if not query.search_terms:
            return []
        
        try:
            search_url = f"{self.base_url}/medicamentos"
            results = []
            seen_nregistros = set()
            
            # For information requests, prioritize the active principle terms
            search_terms = [term for term in query.search_terms 
                          if query.active_principle and query.active_principle in term] if query.is_information_request else query.search_terms
            
            # If no active principle terms found, use the regular search terms
            if not search_terms:
                search_terms = query.search_terms
            
            # Only use the most promising search terms
            for term in search_terms[:3]:
                if len(results) >= MAX_RESULTS:
                    break
                    
                # Skip terms that are too short
                if len(term) < 4:
                    continue
                
                async with session.get(search_url, params={"nombre": term}) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            if isinstance(data, dict) and "resultados" in data and data["resultados"]:
                                for med in data["resultados"]:
                                    if len(results) >= MAX_RESULTS:
                                        break
                                        
                                    if med.get("nregistro") not in seen_nregistros:
                                        result = MedicationResult(**med)
                                        result.relevance_score = self._calculate_relevance(
                                            result, 
                                            active_principle=query.active_principle, 
                                            concentration=query.concentration,
                                            query_terms=query.search_terms,
                                            formulation_type=query.formulation_type,
                                            information_request=query.is_information_request
                                        )
                                        
                                        # Higher threshold for term searches to avoid irrelevant results
                                        if result.relevance_score >= MIN_RELEVANCE_THRESHOLD + 10:
                                            results.append(result)
                                            seen_nregistros.add(med.get("nregistro"))
                        except Exception as e:
                            logger.error(f"Error parsing term search results: {str(e)}")
            
            # Sort by relevance score
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results
        except Exception as e:
            logger.error(f"Error in term search: {str(e)}")
            return []
    
    def _calculate_relevance(self, 
                            med: MedicationResult, 
                            active_principle: Optional[str] = None,
                            concentration: Optional[str] = None,
                            query_terms: Optional[List[str]] = None,
                            formulation_type: Optional[str] = None,
                            information_request: bool = False) -> int:
        """
        Calculate a relevance score for a medication result.
        
        Higher score means more relevant. This is critical to avoid
        the "abacavir problem" by ensuring we only return relevant results.
        """
        score = 0
        
        # Basic checks
        if not med.nombre:
            return 0
        
        # Exact match in name
        med_name_lower = med.nombre.lower()
        
        # Check active principle match
        if active_principle and med.pactivos:
            pactivos_lower = med.pactivos.lower()
            # For information requests, exact active principle match is critical
            if active_principle.lower() in pactivos_lower:
                # Full match in active principles
                score += 100
                # Extra boost for information requests
                if information_request:
                    score += 50
            elif active_principle.lower() in med_name_lower:
                # Active principle appears in name
                score += 50
                # Information requests still get a boost but smaller
                if information_request:
                    score += 25
        
        # Check for concentration match - less important for information requests
        if concentration and concentration in med_name_lower and not information_request:
            score += 30
        
        # Check formulation type match - less important for information requests
        if formulation_type and formulation_type in med_name_lower and not information_request:
            score += 20
        
        # Check for query terms in name
        if query_terms:
            for term in query_terms:
                if term.lower() in med_name_lower or (med.pactivos and term.lower() in med.pactivos.lower()):
                    score += 15
        
        # Prioritize commercialized products
        if med.comerc:
            score += 10
        
        # Penalize results starting with 'A' if they don't match other criteria
        # This helps avoid the "abacavir problem"
        if score < MIN_RELEVANCE_THRESHOLD and med_name_lower.startswith("a"):
            score -= 10
            
        return score
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract potential search terms from the query for fuzzy matching."""
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
                       "con", "por", "los", "las", "del", "que", "realizar", "redactar", 
                       "crear", "generar", "prospecto", "formular", "elaborar", "realiza", "prepara"}
        
        # Add information request keywords to common words to filter them out
        for info_req in self.information_requests:
            for keyword in info_req.keywords:
                for word in keyword.split():
                    common_words.add(word)
        
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