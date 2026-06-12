"""
Streamlined medication search implementation to improve relevance of results.

This module provides an improved search approach that resolves issues with 
irrelevant results like the "abacavir problem" (where search returns alphabetical
results starting with 'A' when no good matches are found).
"""

import re
import logging
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from pydantic import BaseModel, Field
import aiohttp
from dataclasses import dataclass, field

from config import Config
from principle_resolver import ActivePrincipleResolver, ResolvedPrinciple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Minimum relevance score for a result to be considered valid
MIN_RELEVANCE_THRESHOLD = 15  # Increased from 10 to be more strict

# Maximum number of medications to return
MAX_RESULTS = 8

# Known medications with their registration numbers for direct lookup
KNOWN_MEDICATIONS = {
    "minoxidil biorga": "78929",
    "minoxidil": "78929",  # Use same nregistro as fallback
    "biorga": "78929",     # Another variant that might be used
    "regaine": "81897",    # Another minoxidil brand
    "ibuprofeno": "43513", # Example of a common medication
    "paracetamol": "64033" # Example of a common medication
}

@dataclass
class QueryIntent:
    """Represents the intent of a medication-related query."""
    intent_type: str  # Type of intent (e.g., "general", "contraindications", "dosage", etc.)
    description: str  # Human-readable description of the intent
    section_key: Optional[str] = None  # Optional section key to prioritize

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
    exact_medication_matches: List[str] = Field(default_factory=list)  # New field for direct known medication matches
    idpractiv1: Optional[int] = None  # Official CIMA active-principle id (resolved via maestras)
    resolved_principle_name: Optional[str] = None  # Official catalogue name for the resolved principle
    
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
    principle_resolver: ActivePrincipleResolver = field(default_factory=ActivePrincipleResolver)
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
"vitamina d", "calcio", "hierro", "insulina", "metronidazol", "minoxidil",
"nolotil", "escitalopram", "bromazepam", "pantoprazol", "citalopram",
"esomeprazol", "sertralina", "bisoprolol", "olmesartan", "rosuvastatina",
"duloxetina", "clopidogrel", "furosemida", "ramipril", "paroxetina",
"micofenolato", "olanzapina", "lansoprazol", "irbesartan", "nebivolol",
"torasemida", "pregabalina", "venlafaxina", "gabapentina", "carvedilol",
"tamsulosina", "telmisartan", "metoclopramida", "levocetirizina",
"dexketoprofeno", "deflazacort", "mirtazapina", "ebastina", "propranolol",
"candesartan", "sildenafilo", "tacrolimus", "ezetimiba", "levonorgestrel",
"raltegravir", "donepezilo", "fexofenadina", "clortalidona", "trazodona",
"levetiracetam", "solifenacina", "rivaroxaban", "glimepirida", "memantina",
"biorga"  # Added as a potential active principle for minoxidil biorga
        
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
    
    async def execute_search(self, query_text: str) -> Tuple[List[Dict[str, Any]], str, Optional[QueryIntent]]:
        """
        Execute a comprehensive search for medications based on the query.
        
        Args:
            query_text: The search query
            
        Returns:
            Tuple[List[Dict], str, Optional[QueryIntent]]: List of results, quality assessment, and query intent
        """
        session = None
        quality = "unknown"
        query_intent = None
        
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
            
            # Analyze the query - IMPROVED to better detect medications
            query_info = self._analyze_query(query_text)
            logger.info(f"Query analysis: Active principle: {query_info.active_principle}, Information request: {query_info.is_information_request}, Type: {query_info.information_request_type}")
            logger.info(f"Exact medication matches: {query_info.exact_medication_matches}, Uppercase names: {query_info.uppercase_names}")

            # Resolve the detected term(s) against the official CIMA active-principle
            # catalogue (maestras?maestra=1). A successful resolution gives us the
            # official idpractiv1, enabling exact-id retrieval that eliminates the
            # fuzzy name matching (the root cause of the "abacavir problem") and
            # removes the dependency on the hardcoded principle list: any principle
            # in the official catalogue is recognised, not just the ~100 local ones.
            resolution_candidates: List[str] = []
            if query_info.active_principle:
                resolution_candidates.append(query_info.active_principle)
            resolution_candidates.extend(query_info.uppercase_names)
            resolution_candidates.extend(query_info.search_terms)

            resolved = await self.principle_resolver.resolve_candidates(session, resolution_candidates)
            if resolved:
                query_info.idpractiv1 = resolved.id
                query_info.resolved_principle_name = resolved.nombre
                # Use the official catalogue name for downstream scoring/gating so
                # relevance checks compare against canonical spelling.
                query_info.active_principle = resolved.nombre.lower()

            # Create query intent if this is an information request
            if query_info.is_information_request and query_info.information_request_type:
                # Map information request types to section keys and descriptions
                section_key_map = {
                    "contraindications": ("contraindicaciones", "contraindicaciones"),
                    "side_effects": ("efectos_adversos", "efectos adversos"),
                    "dosage": ("posologia_procedimiento", "posología y administración"),
                    "interactions": ("interacciones", "interacciones"),
                    "precautions": ("advertencias", "advertencias y precauciones"),
                    "indications": ("indicaciones", "indicaciones terapéuticas"),
                    "administration": ("posologia_procedimiento", "forma de administración"),
                    "composition": ("composicion", "composición"),
                    "conservation": ("conservacion", "conservación")
                }
                
                # Get section key and description for this information request type
                section_key, description = section_key_map.get(
                    query_info.information_request_type, 
                    (None, query_info.information_request_type)
                )
                
                # Create query intent
                query_intent = QueryIntent(
                    intent_type=query_info.information_request_type,
                    description=description,
                    section_key=section_key
                )
            else:
                # Create a general query intent
                query_intent = QueryIntent(
                    intent_type="general",
                    description="información general",
                    section_key=None
                )
            
            # Results storage
            all_results = []
            seen_nregistros = set()
            
            # NEW! First try: direct lookup for known medications
            # This is the most important improvement to fix the "abacavir problem"
            if query_info.exact_medication_matches:
                logger.info(f"Trying direct lookup for known medications: {query_info.exact_medication_matches}")
                direct_results = await self._direct_medication_lookup(session, query_info)
                
                if direct_results:
                    logger.info(f"Found direct results for known medication: {len(direct_results)} results")
                    
                    # Add all direct results
                    all_results.extend(direct_results)
                    seen_nregistros.update([r.nregistro for r in direct_results])
                    
                    # Direct lookups are high quality by definition
                    quality = "very_high"
                    
                    # Return immediately if we have high confidence direct results
                    if len(direct_results) >= 1:
                        # Sort by relevance and filter low-relevance results
                        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
                        
                        # Convert to dictionaries for easier integration
                        return [result.dict() for result in all_results], quality, query_intent
            
            # PRIORITY: exact retrieval by official active-principle id. This is the
            # most reliable strategy after direct nregistro lookups: the API filters
            # by identifier, so no alphabetical filler can leak into the results.
            if query_info.idpractiv1 and len(all_results) < MAX_RESULTS:
                logger.info(f"Searching by official principle id: {query_info.idpractiv1} ({query_info.resolved_principle_name})")
                id_results = await self._search_by_principle_id(session, query_info)

                for result in id_results:
                    if result.nregistro not in seen_nregistros:
                        all_results.append(result)
                        seen_nregistros.add(result.nregistro)

                if id_results and quality == "unknown":
                    quality = "very_high"

            # For information requests: prioritize exact matches on active principle
            # (name-based fallback when the catalogue resolution did not succeed)
            if (query_info.is_information_request and query_info.active_principle
                    and not query_info.idpractiv1):
                logger.info(f"Processing information request about {query_info.active_principle}")
                info_results = await self._search_by_active_principle(session, query_info, prioritize_exact_match=True)

                # Add results
                all_results.extend(info_results)
                seen_nregistros.update([r.nregistro for r in info_results])

                if info_results:
                    quality = "high"

            # Next, search for uppercase medication names like "MINOXIDIL BIORGA"
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
            
            # Next, search by active principle NAME — only as fallback when the
            # official catalogue resolution did not yield an id.
            if len(all_results) < MAX_RESULTS and query_info.active_principle and not query_info.idpractiv1:
                logger.info(f"Searching by active principle: {query_info.active_principle}")
                ap_results = await self._search_by_active_principle(session, query_info)
                
                # Add unique results
                for result in ap_results:
                    if result.nregistro not in seen_nregistros:
                        all_results.append(result)
                        seen_nregistros.add(result.nregistro)
                
                if ap_results and quality == "unknown":
                    quality = "medium"
            
            # Next: search by full name
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
            
            # Last resort: search by individual terms
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

            # Content-match gate: keep only results that genuinely match the query
            # (active principle, term or brand). This solves the "abacavir problem"
            # (irrelevant alphabetical fillers passing the score threshold) for every
            # letter, instead of the old alphabetical 'A' penalty that wrongly
            # punished legitimate A-drugs. If nothing passes the gate (e.g. very
            # broad queries) we fall back to the score-filtered list.
            gated_results = [r for r in filtered_results if self._has_content_match(r, query_info)]
            if gated_results:
                filtered_results = gated_results

            # Limit to maximum results
            filtered_results = filtered_results[:MAX_RESULTS]
            
            # Set quality to no_results if we didn't find anything
            if not filtered_results:
                quality = "no_results"
            
            logger.info(f"Search completed: {len(filtered_results)} results with quality {quality}")
            
            # Convert to dictionaries for easier integration
            return [result.dict() for result in filtered_results], quality, query_intent
            
        except Exception as e:
            logger.error(f"Error executing search: {str(e)}")
            return [], "error", query_intent
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
        
        # NEW! Check for direct matches with known medications first
        exact_medication_matches = []
        for med_name in KNOWN_MEDICATIONS.keys():
            if med_name in query_lower:
                exact_medication_matches.append(med_name)
        
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
        
        # IMPROVED! Extract uppercase medication names with more patterns
        uppercase_names = []
        
        # Standard pattern for uppercase names like "MINOXIDIL BIORGA"
        standard_uppercase = re.findall(r'\b[A-Z]{2,}\s+[A-Z]{2,}\b', query_text)
        if standard_uppercase:
            uppercase_names.extend(standard_uppercase)
        
        # If no standard uppercase names found, try looking for specific brands in any case
        if not uppercase_names:
            # Custom pattern for specific brand names (case insensitive)
            brand_pattern = r'(?i)\b(minoxidil\s+biorga|biorga\s+minoxidil)\b'
            brand_matches = re.findall(brand_pattern, query_text)
            
            uppercase_names.extend([match.upper() for match in brand_matches])
        
        # Extract active principle - this is critical for information requests
        active_principle = None
        
        # First look for known active principles - sorted by length to prioritize longest matches
        principles_by_length = sorted(self.active_principles, key=len, reverse=True)
        for ap in principles_by_length:
            if ap in query_lower:
                active_principle = ap
                break
        
        # If still not found, try other extraction methods
        if not active_principle:
            # Check if we have exact medication matches and use their active principles
            if exact_medication_matches:
                # For now, just use the medication name as the active principle
                active_principle = exact_medication_matches[0]
            else:
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
                        # Use uppercase name as active principle if found
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
        
        # IMPROVED! Extract search terms with better filtering
        search_terms = self._extract_search_terms(query_text)
        
        # Check if this is a prospecto request
        prospecto_pattern = r'(?:redactar|generar|crear|elaborar|realizar?e?|escrib[ei]r|hac[ae]r|desarroll[ae]r|realiza(?:r|)|prepar(?:ar|a))\s+(?:un|el|uns?|una?)?\s+prospecto'
        is_prospecto = bool(re.search(prospecto_pattern, query_lower))
        
        # Create and return the query object with our new exact_medication_matches field
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
            information_request_type=information_request_type,
            exact_medication_matches=exact_medication_matches
        )
    
    async def _direct_medication_lookup(self, session: aiohttp.ClientSession, query: MedicationQuery) -> List[MedicationResult]:
        """
        NEW! Direct lookup for known medications by registration number.
        This is a key improvement to solve the "abacavir problem".
        """
        if not query.exact_medication_matches:
            return []
        
        results = []
        
        # Try direct lookup for all matched medications
        for med_name in query.exact_medication_matches:
            if med_name not in KNOWN_MEDICATIONS:
                continue
                
            nregistro = KNOWN_MEDICATIONS[med_name]
            medication_url = f"{self.base_url}/medicamento"
            
            try:
                # First try direct lookup by registration number - most reliable
                logger.info(f"Attempting direct lookup for {med_name} with nregistro: {nregistro}")
                async with session.get(medication_url, params={"nregistro": nregistro}) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            if data:
                                # Ensure it's the correct format
                                if "nregistro" in data and "nombre" in data:
                                    result = MedicationResult(**data)
                                    result.relevance_score = 150  # Very high score for direct lookups
                                    results.append(result)
                                    logger.info(f"Direct lookup successful for {med_name}: {data.get('nombre', 'Unknown')}")
                                else:
                                    logger.warning(f"Direct lookup returned data in unexpected format: {data}")
                        except Exception as e:
                            logger.error(f"Error parsing response for direct lookup of {med_name}: {str(e)}")
                    else:
                        logger.warning(f"Direct lookup failed with status {response.status} for {med_name}")
                        
                # If direct lookup failed, try medicamentos search as fallback
                if not results:
                    search_url = f"{self.base_url}/medicamentos"
                    logger.info(f"Trying medicamentos search for {med_name} as fallback")
                    
                    # Try both by nombre and by nregistro
                    async with session.get(search_url, params={"nombre": med_name.upper()}) as response:
                        if response.status == 200:
                            try:
                                data = await response.json()
                                if isinstance(data, dict) and "resultados" in data and data["resultados"]:
                                    for med in data["resultados"]:
                                        # Check if the result has the right registration number
                                        if med.get("nregistro") == nregistro:
                                            result = MedicationResult(**med)
                                            result.relevance_score = 130  # High score but not as high as direct lookup
                                            results.append(result)
                                            logger.info(f"Found {med_name} using nombre search")
                                            break
                            except Exception as e:
                                logger.error(f"Error in nombre search fallback for {med_name}: {str(e)}")
            except Exception as e:
                logger.error(f"Error in direct lookup for {med_name}: {str(e)}")
        
        return results
    
    async def _lookup_by_nregistro(self, session: aiohttp.ClientSession, nregistro: str,
                                   relevance_score: int = 130) -> List[MedicationResult]:
        """
        Resolve a medication by its registration number via the real CIMA API.

        Used instead of fabricating MedicationResult objects locally, so that the
        name, active principles and marketing-authorisation holder always reflect
        the official CIMA data rather than a hardcoded (and possibly wrong) value.
        Returns [] if the API does not return a usable record.
        """
        medication_url = f"{self.base_url}/medicamento"
        try:
            async with session.get(medication_url, params={"nregistro": nregistro}) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, dict) and data.get("nregistro") and data.get("nombre"):
                        result = MedicationResult(**data)
                        result.relevance_score = relevance_score
                        return [result]
                    logger.warning(f"Lookup for nregistro {nregistro} returned unexpected data")
                else:
                    logger.warning(f"Lookup for nregistro {nregistro} failed with status {response.status}")
        except Exception as e:
            logger.error(f"Error looking up nregistro {nregistro}: {str(e)}")
        return []

    async def _search_by_principle_id(self, session: aiohttp.ClientSession,
                                      query: MedicationQuery) -> List[MedicationResult]:
        """
        Exact retrieval by official active-principle id (idpractiv1), as documented
        in the CIMA REST API ("GET medicamentos?{condiciones}"). The API itself
        guarantees every result contains the principle, so no alphabetical filler
        can appear; scoring is only used to order presentations.
        """
        if not query.idpractiv1:
            return []

        search_url = f"{self.base_url}/medicamentos"
        params = {"idpractiv1": str(query.idpractiv1), "pagina": "1"}
        headers = {"Accept": "application/json"}
        results: List[MedicationResult] = []
        seen: Set[str] = set()

        try:
            async with session.get(search_url, params=params, headers=headers) as response:
                if response.status != 200:
                    logger.warning(f"idpractiv1 search returned status {response.status}")
                    return []
                data = await response.json()
        except Exception as e:
            logger.error(f"Error in idpractiv1 search: {str(e)}")
            return []

        if not isinstance(data, dict) or not data.get("resultados"):
            return []

        for med in data["resultados"]:
            if not isinstance(med, dict) or med.get("nregistro") in seen:
                continue
            try:
                result = MedicationResult(**med)
            except Exception as e:
                logger.warning(f"Skipping malformed idpractiv1 result: {str(e)}")
                continue

            # Base score reflects the exact-id provenance; bonuses order the list
            # by how well each presentation matches the rest of the query.
            score = 110
            name_lower = (result.nombre or "").lower()
            # Compare concentrations ignoring spacing ("250mg/5ml" vs "250 mg/5 ml")
            if query.concentration:
                conc_compact = query.concentration.lower().replace(" ", "")
                if conc_compact in name_lower.replace(" ", ""):
                    score += 30
            if query.formulation_type:
                keywords = Config.FORMULATION_TYPES.get(query.formulation_type, [])
                if any(keyword in name_lower for keyword in keywords):
                    score += 20
            if result.comerc:
                score += 20

            result.relevance_score = score
            results.append(result)
            seen.add(result.nregistro)

            if len(results) >= MAX_RESULTS * 2:  # Keep a buffer before final sort
                break

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:MAX_RESULTS]

    async def _search_by_uppercase(self, session: aiohttp.ClientSession, query: MedicationQuery) -> List[MedicationResult]:
        """IMPROVED! Search for exact matches with uppercase medication names."""
        if not query.uppercase_names:
            return []
        
        try:
            uppercase_name = query.uppercase_names[0]
            search_url = f"{self.base_url}/medicamentos"
            results = []
            
            # Define search approaches in order of preference
            search_approaches = [
                {"params": {"nombre": uppercase_name}, "desc": "exact uppercase"},
                {"params": {"nombre": uppercase_name.replace(" ", "+")}, "desc": "plus-separated"},
                {"params": {"nombre": uppercase_name.split()[0]}, "desc": "first word"} if len(uppercase_name.split()) > 1 else None
            ]
            
            # Filter out None entries
            search_approaches = [approach for approach in search_approaches if approach]
            
            # Try each approach
            for approach in search_approaches:
                if len(results) >= MAX_RESULTS:
                    break
                
                params = approach["params"]
                desc = approach["desc"]
                
                try:
                    logger.info(f"Trying uppercase search with {desc} approach: {params}")
                    async with session.get(search_url, params=params) as response:
                        if response.status == 200:
                            try:
                                data = await response.json()
                                if isinstance(data, dict) and "resultados" in data and data["resultados"]:
                                    for med in data["resultados"]:
                                        result = MedicationResult(**med)
                                        
                                        # Highly relevant if name matches exactly (case insensitive)
                                        if uppercase_name.lower() in result.nombre.lower():
                                            result.relevance_score = 120
                                        # Less relevant if it only contains part of the name
                                        elif any(word.lower() in result.nombre.lower() for word in uppercase_name.split()):
                                            result.relevance_score = 90
                                        else:
                                            result.relevance_score = 70
                                        
                                        # Only add if above threshold - more strict for uppercase searches
                                        if result.relevance_score >= MIN_RELEVANCE_THRESHOLD + 5:
                                            results.append(result)
                                            
                                    logger.info(f"Found {len(results)} results with {desc} approach")
                                    
                                    # If we found good results, break early
                                    if len(results) >= 1 and any(r.relevance_score >= 100 for r in results):
                                        break
                            except Exception as e:
                                logger.error(f"Error parsing uppercase search results with {desc} approach: {str(e)}")
                except Exception as e:
                    logger.error(f"Error in uppercase search with {desc} approach: {str(e)}")
            
            # Sort by relevance
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results[:MAX_RESULTS]
        except Exception as e:
            logger.error(f"Error in uppercase search: {str(e)}")
            return []
    
    async def _search_by_active_principle(self, session: aiohttp.ClientSession, query: MedicationQuery, prioritize_exact_match: bool = False) -> List[MedicationResult]:
        """IMPROVED! Search by active principle with better handling of special cases."""
        if not query.active_principle:
            return []
        
        try:
            active_principle = query.active_principle

            # Special case handling for Minoxidil Biorga: resolve via the real CIMA
            # API instead of fabricating name/lab/nregistro locally. Returning
            # hardcoded medical data is unsafe (it may be wrong and is presented
            # authoritatively); if the lookup fails we fall through to normal search.
            if "minoxidil" in active_principle.lower() or "biorga" in active_principle.lower():
                logger.info("Special case: resolving minoxidil/biorga via CIMA API")
                api_results = await self._lookup_by_nregistro(session, "78929", relevance_score=150)
                if api_results:
                    return api_results

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

                # "practiv1" is the documented parameter for active-principle name
                # search (CIMA REST API v1.23); the previously used
                # "principiosActivos" does not exist in the API.
                search_params = [
                    {"practiv1": variation, "desc": "practiv1"}
                ]

                for params_dict in search_params:
                    desc = params_dict["desc"]
                    params = {"practiv1": variation, "pagina": "1"}

                    try:
                        logger.info(f"Trying active principle search with {desc} approach: {params}")
                        async with session.get(search_url, params=params) as response:
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

                                                # Only add if above threshold
                                                if result.relevance_score >= MIN_RELEVANCE_THRESHOLD:
                                                    results.append(result)
                                                    seen_nregistros.add(med.get("nregistro"))
                                except Exception as e:
                                    logger.error(f"Error parsing active principle search results with {desc} approach: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error in active principle search with {desc} approach: {str(e)}")
            
            # Special handling for minoxidil: if nothing matched, resolve via the
            # real CIMA API rather than fabricating a result.
            if "minoxidil" in active_principle.lower() and not results:
                api_results = await self._lookup_by_nregistro(session, "78929", relevance_score=130)
                results.extend(api_results)

            # Sort by relevance score
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results
        except Exception as e:
            logger.error(f"Error in active principle search: {str(e)}")
            return []
    
    async def _search_by_name(self, session: aiohttp.ClientSession, query: MedicationQuery) -> List[MedicationResult]:
        """IMPROVED! Search by complete name with better filtering of irrelevant results."""
        try:
            search_url = f"{self.base_url}/medicamentos"
            results = []
            seen_nregistros = set()
            
            # For information requests, just search by active principle if available
            search_term = query.active_principle if query.is_information_request and query.active_principle else query.query_text
            
            # Special case for queries containing uppercase names like MINOXIDIL BIORGA
            if query.uppercase_names:
                search_term = query.uppercase_names[0]
            
            logger.info(f"Searching by name using term: {search_term}")
            
            # Try multiple search variations for better results
            variations = [
                search_term,
                search_term.replace(" ", "+"),
                " ".join(search_term.split()[:2]) if len(search_term.split()) > 2 else None  # First two words
            ]
            
            # Filter out None entries
            variations = [var for var in variations if var]
            
            for var in variations:
                if len(results) >= MAX_RESULTS:
                    break
                    
                try:
                    async with session.get(search_url, params={"nombre": var}) as response:
                        if response.status == 200:
                            try:
                                data = await response.json()
                                if isinstance(data, dict) and "resultados" in data and data["resultados"]:
                                    for med in data["resultados"]:
                                        if med.get("nregistro") not in seen_nregistros:
                                            result = MedicationResult(**med)
                                            
                                            # Calculate relevance score
                                            result.relevance_score = self._calculate_relevance(
                                                result, 
                                                active_principle=query.active_principle, 
                                                concentration=query.concentration,
                                                query_terms=query.search_terms,
                                                formulation_type=query.formulation_type,
                                                information_request=query.is_information_request
                                            )
                                            
                                            # Only add if it genuinely matches the query
                                            # by content and clears the score threshold.
                                            if (result.relevance_score >= MIN_RELEVANCE_THRESHOLD
                                                    and self._has_content_match(result, query)):
                                                results.append(result)
                                                seen_nregistros.add(med.get("nregistro"))
                            except Exception as e:
                                logger.error(f"Error parsing name search results: {str(e)}")
                except Exception as e:
                    logger.error(f"Error in name search for variation {var}: {str(e)}")

            # Sort by relevance score
            results.sort(key=lambda x: x.relevance_score, reverse=True)

            # Limit to max results
            return results[:MAX_RESULTS]
        except Exception as e:
            logger.error(f"Error in name search: {str(e)}")
            return []
    
    async def _search_by_terms(self, session: aiohttp.ClientSession, query: MedicationQuery) -> List[MedicationResult]:
        """IMPROVED! Search by individual terms with better filtering for the abacavir problem."""
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
            
            # Use only the most promising search terms
            # Sort by length first to prioritize longer terms
            search_terms = sorted(search_terms, key=len, reverse=True)
            
            for term in search_terms[:3]:  # Only use top 3 terms
                if len(results) >= MAX_RESULTS:
                    break
                    
                # Skip terms that are too short or common
                if len(term) < 4 or term.lower() in ["para", "como", "sobre", "este", "esta"]:
                    continue
                
                # Skip common medication-unrelated words
                if term.lower() in ["información", "información", "buscar", "encontrar"]:
                    continue
                
                try:
                    logger.info(f"Searching by term: {term}")
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
                                            
                                            # Calculate relevance score with higher threshold for term searches
                                            result.relevance_score = self._calculate_relevance(
                                                result, 
                                                active_principle=query.active_principle, 
                                                concentration=query.concentration,
                                                query_terms=query.search_terms,
                                                formulation_type=query.formulation_type,
                                                information_request=query.is_information_request
                                            )
                                            
                                            # Higher threshold for term searches, and
                                            # require a genuine content match (replaces
                                            # the old alphabetical 'A' penalty).
                                            if (result.relevance_score >= MIN_RELEVANCE_THRESHOLD + 10
                                                    and self._has_content_match(result, query)):
                                                results.append(result)
                                                seen_nregistros.add(med.get("nregistro"))
                            except Exception as e:
                                logger.error(f"Error parsing term search results: {str(e)}")
                except Exception as e:
                    logger.error(f"Error in term search for {term}: {str(e)}")
            
            # Sort by relevance score
            results.sort(key=lambda x: x.relevance_score, reverse=True)

            return results[:MAX_RESULTS]
        except Exception as e:
            logger.error(f"Error in term search: {str(e)}")
            return []
    
    def _has_content_match(self, med: MedicationResult, query: MedicationQuery) -> bool:
        """
        Whether a result genuinely matches the query by content (active principle,
        search term or uppercase brand appearing in the name or active principles).

        This replaces the old "penalise everything starting with 'A'" heuristic used
        to work around the "abacavir problem". That heuristic was wrong: it punished
        legitimate drugs (amoxicilina, atorvastatina, aspirina, azitromicina,
        amlodipino, alprazolam, ...). Requiring a real content match fixes the
        irrelevant-alphabetical-result problem for ALL letters without bias.
        """
        name = (med.nombre or "").lower()
        pact = (med.pactivos or "").lower()

        # Results retrieved via the official idpractiv1 filter are relevant by
        # construction (the API guarantees the principle); when the list payload
        # omits pactivos we must not refute that guarantee here.
        if query.idpractiv1 and not pact:
            return True

        ap = (query.active_principle or "").lower()
        if ap and (ap in name or ap in pact):
            return True

        for term in (query.search_terms or []):
            t = term.lower()
            if len(t) >= 4 and (t in name or t in pact):
                return True

        for up in (query.uppercase_names or []):
            if any(len(w) >= 3 and w.lower() in name for w in up.split()):
                return True

        return False

    def _calculate_relevance(self, med: MedicationResult, active_principle: Optional[str] = None,
                             concentration: Optional[str] = None, query_terms: Optional[List[str]] = None, 
                             formulation_type: Optional[str] = None, information_request: bool = False) -> int:
        """IMPROVED! Calculate medication relevance score with better filtering for the abacavir problem."""
        score = 0
        
        # Basic checks
        if not med.nombre:
            return 0
        
        med_name_lower = med.nombre.lower()

        # Check active principle match - highest priority
        if active_principle and med.pactivos:
            pactivos_lower = med.pactivos.lower()

            # Full match in active principles - most important factor
            if active_principle.lower() in pactivos_lower:
                score += 100
            # Active principle appears in name
            elif active_principle.lower() in med_name_lower:
                score += 50
        
        # Check for concentration match
        if concentration and concentration in med_name_lower:
            score += 30
        
        # Check for formulation type match
        if formulation_type:
            formulation_types = Config.FORMULATION_TYPES
            if formulation_type in formulation_types:
                keywords = formulation_types[formulation_type]
                if any(keyword in med_name_lower for keyword in keywords):
                    score += 20
        
        # Check for query terms in name or active principles
        if query_terms:
            for term in query_terms:
                term_lower = term.lower()
                # Direct match in name
                if term_lower in med_name_lower:
                    score += 20
                # Match in active principles
                elif med.pactivos and term_lower in med.pactivos.lower():
                    score += 15
                # Partial match (for compounds)
                elif len(term) > 4 and any(term_lower[:5] in word for word in med_name_lower.split()):
                    score += 10
        
        # Prioritize commercialized products
        if med.comerc:
            score += 20
        
        # Special case handling for minoxidil/biorga
        if ("minoxidil" in med_name_lower or "biorga" in med_name_lower) and med.nregistro == "78929":
            score += 50  # Big boost for this specific product

        return score
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """IMPROVED! Extract potential search terms from the query with better filtering."""
        # Patterns for potential medication names and active ingredients
        patterns = [
            r'([A-Z][a-záéíóúñ]+(?:\s[a-záéíóúñ]+){0,3})',  # Capitalized words
            r'(\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|UI|unidades))',  # Dosages
            r'([A-Za-záéíóúñ]+\+[A-Za-záéíóúñ]+)',  # Combinations with +
            r'(?i)(minoxidil\s+biorga|biorga\s+minoxidil)'  # Specific case for Minoxidil Biorga
        ]
        
        # Extract terms using all patterns
        potential_terms = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            potential_terms.extend([m.strip() for m in matches if len(m.strip()) > 3])
        
        # Add individual words that might be medication names
        # Expanded list of common words to filter out
        common_words = {
            "sobre", "para", "como", "este", "esta", "estos", "estas", "cual", "cuales", 
            "con", "por", "los", "las", "del", "que", "realizar", "realizar", "redactar", 
            "crear", "generar", "prospecto", "formular", "elaborar", "realiza", "prepara",
            "información", "informacion", "buscar", "encontrar", "consultar", "ver", "saber",
            "quiero", "necesito", "puedo", "debe", "puede", "debo", "cómo", "como", "cuando",
            "dónde", "donde", "qué", "que", "quién", "quien", "cuándo", "cuando", "cuánto", "cuanto"
        }
        
        # Add information request keywords to filter list
        info_request_keywords = [
            "contraindicaciones", "contraindicación", "no tomar", "no usar", "no debe",
            "efectos secundarios", "efectos adversos", "reacciones adversas", "adversos",
            "posología", "posologia", "dosis", "dosificación", "como tomar", "como usar",
            "interacciones", "interacción", "junto con", "combinado con", "mezclar con",
            "precauciones", "advertencias", "cuidados", "tenga cuidado", "atención",
            "indicado para", "indicaciones", "uso", "para qué", "para que", "para qué sirve",
            "administración", "administracion", "vía", "via", "modo de empleo", "como administrar",
            "composición", "composicion", "ingredientes", "componentes", "excipientes",
            "conservación", "conservacion", "almacenamiento", "guardar", "caducidad"
        ]
        
        # Add all keywords to common words
        for keyword in info_request_keywords:
            for word in keyword.split():
                common_words.add(word.lower())
        
        # Extract individual words
        words = query.split()
        for word in words:
            word_lower = word.lower()
            if len(word) > 4 and word_lower not in common_words and word not in potential_terms:
                potential_terms.append(word)
        
        # Add bi-grams (pairs of words) - these can be helpful for finding compound names
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i+1]) > 3:
                bigram = f"{words[i]} {words[i+1]}"
                bigram_lower = bigram.lower()
                # Skip if both words are common
                if words[i].lower() in common_words and words[i+1].lower() in common_words:
                    continue
                if bigram not in potential_terms:
                    potential_terms.append(bigram)
        
        # Check for active principles in database
        for ap in self.active_principles:
            if ap in query.lower() and ap not in potential_terms:
                potential_terms.append(ap)
        
        # Eliminate duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in potential_terms:
            if term.lower() not in seen:
                unique_terms.append(term)
                seen.add(term.lower())
        
        return unique_terms
