"""
LangGraph implementation for improved medication search in CIMA Assistant.

This module provides a structured approach to search for medications using a graph-based workflow
with validation through Pydantic models. It solves issues with irrelevant results (like returning
abacavir for unrelated queries) by implementing multiple search strategies and relevance checks.
"""

import re
import logging
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
import aiohttp
from dataclasses import dataclass, field

from config import Config

# Configure logging
logger = logging.getLogger(__name__)

# Minimum relevance score for a result to be considered valid
MIN_RELEVANCE_THRESHOLD = 20

# Maximum number of medications to return
MAX_RESULTS = 5

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
    
class SearchResults(BaseModel):
    """Container for search results with metadata."""
    results: List[MedicationResult] = Field(default_factory=list)
    query: MedicationQuery
    search_method: str
    successful: bool = True

class SearchState(BaseModel):
    """The complete state of the search workflow."""
    query: MedicationQuery
    exact_matches: Optional[SearchResults] = None
    active_principle_matches: Optional[SearchResults] = None
    fuzzy_matches: Optional[SearchResults] = None
    name_matches: Optional[SearchResults] = None
    final_results: List[MedicationResult] = Field(default_factory=list)
    result_quality: str = "unknown"
    errors: List[str] = Field(default_factory=list)
    session: Optional[Any] = None  # aiohttp.ClientSession can't be serialized in Pydantic

@dataclass
class MedicationSearchGraph:
    """
    Graph-based workflow for medication searches.
    
    This class implements a structured search approach using LangGraph
    to find medications in CIMA API with improved relevance filtering.
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
    
    def __post_init__(self):
        """Initialize the graph after initialization."""
        self.workflow = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the search workflow graph."""
        # Create the graph
        workflow = StateGraph(StateGraph.from_pydantic(SearchState))
        
        # Add nodes
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("search_uppercase_exact_match", self.search_uppercase_exact_match)
        workflow.add_node("search_active_principle", self.search_active_principle)
        workflow.add_node("search_by_name", self.search_by_name)
        workflow.add_node("search_terms", self.search_terms)
        workflow.add_node("evaluate_results", self.evaluate_results)
        
        # Define the edges (flow of execution)
        workflow.add_edge("analyze_query", "search_uppercase_exact_match")
        
        # Add conditional edges from uppercase exact match
        workflow.add_conditional_edges(
            "search_uppercase_exact_match",
            self.check_uppercase_results,
            {
                "found_exact": "evaluate_results",
                "not_found": "search_active_principle"
            }
        )
        
        # Add conditional edges from active principle search
        workflow.add_conditional_edges(
            "search_active_principle",
            self.check_ap_results,
            {
                "found": "evaluate_results",
                "not_found": "search_by_name"
            }
        )
        
        # Add conditional edges from name search
        workflow.add_conditional_edges(
            "search_by_name",
            self.check_name_results,
            {
                "found": "evaluate_results",
                "not_found": "search_terms"
            }
        )
        
        # Add edge from search_terms to evaluate_results
        workflow.add_edge("search_terms", "evaluate_results")
        
        # Set entry and final nodes
        workflow.add_edge("evaluate_results", END)
        
        # Compile the graph
        return workflow.compile()
    
    async def analyze_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the query to extract structured search parameters.
        
        This node extracts key components from the text query:
        - Active principle
        - Formulation type
        - Administration route
        - Concentration
        - Uppercase medication names
        """
        query_text = state["query"].query_text
        query_lower = query_text.lower()
        
        # Extract uppercase medication names
        uppercase_names = re.findall(r'\b[A-Z]{2,}\s+[A-Z]{2,}\b', query_text.upper())
        
        # Extract active principle
        active_principle = state["query"].active_principle
        if not active_principle:
            # Look for known active principles
            for ap in self.active_principles:
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
                        # Just take the longest word as a guess
                        words = [w for w in query_lower.split() if len(w) > 4 and not any(x in w for x in ['como', 'para', 'sobre', 'cual', 'este', 'esta'])]
                        if words:
                            active_principle = max(words, key=len)
        
        # Extract formulation type
        formulation_type = state["query"].formulation_type
        if not formulation_type:
            formulation_types = Config.FORMULATION_TYPES
            for form_type, keywords in formulation_types.items():
                if any(word in query_lower for word in keywords):
                    formulation_type = form_type
                    break
        
        # Extract administration route
        admin_route = state["query"].administration_route
        if not admin_route:
            admin_routes = Config.ADMIN_ROUTES
            for route, keywords in admin_routes.items():
                if any(word in query_lower for word in keywords):
                    admin_route = route
                    break
        
        # Extract concentration
        concentration = state["query"].concentration
        if not concentration:
            concentration_pattern = r'(\d+(?:[,.]\d+)?\s*(?:%|mg|g|ml|mcg|UI|unidades)|\d+\s*(?:mg)?[/](?:ml|g))'
            concentration_match = re.search(concentration_pattern, query_text)
            if concentration_match:
                concentration = concentration_match.group(0)
        
        # Extract search terms for fuzzy matching
        search_terms = self._extract_search_terms(query_text)
        
        # Check if this is a prospecto request
        prospecto_pattern = r'(?:redactar|generar|crear|elaborar|realizar?e?|escrib[ei]r|hac[ae]r|desarroll[ae]r|realiza(?:r|)|prepar(?:ar|a))\s+(?:un|el|uns?|una?)?\s+prospecto'
        is_prospecto = bool(re.search(prospecto_pattern, query_lower))
        
        # Update the query with extracted information
        updated_query = MedicationQuery(
            query_text=query_text,
            active_principle=active_principle,
            formulation_type=formulation_type,
            administration_route=admin_route,
            concentration=concentration,
            uppercase_names=uppercase_names,
            search_terms=search_terms,
            is_prospecto=is_prospecto
        )
        
        # Initialize session if needed
        session = None
        if "session" not in state or not state["session"]:
            try:
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
            except Exception as e:
                logger.error(f"Error creating session: {str(e)}")
                return {"query": updated_query, "errors": [f"Error creating session: {str(e)}"]}
        else:
            session = state["session"]
        
        logger.info(f"Analyzed query: {updated_query}")
        return {"query": updated_query, "session": session}
    
    async def search_uppercase_exact_match(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for exact matches with uppercase medication names like 'MINOXIDIL BIORGA'.
        
        This is a high-precision search for when medication names are clearly specified.
        """
        query = state["query"]
        session = state["session"]
        
        if not query.uppercase_names:
            # No uppercase names to search for, skip this step
            return {
                "exact_matches": SearchResults(
                    results=[],
                    query=query,
                    search_method="uppercase_exact",
                    successful=True
                )
            }
        
        try:
            # Search for the uppercase name directly
            uppercase_name = query.uppercase_names[0]
            logger.info(f"Searching for uppercase name: {uppercase_name}")
            
            search_url = f"{self.base_url}/medicamentos"
            async with session.get(search_url, params={"nombre": uppercase_name}) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        if isinstance(data, dict) and "resultados" in data and data["resultados"]:
                            results = []
                            seen_nregistros = set()
                            
                            for med in data["resultados"]:
                                if med.get("nregistro") not in seen_nregistros:
                                    result = MedicationResult(**med)
                                    # Give a high relevance score to exact uppercase matches
                                    if uppercase_name.lower() in result.nombre.lower():
                                        result.relevance_score = 100
                                    else:
                                        # Basic relevance - we'll refine this
                                        result.relevance_score = 80
                                    
                                    results.append(result)
                                    seen_nregistros.add(med.get("nregistro"))
                            
                            logger.info(f"Found {len(results)} exact uppercase matches")
                            return {
                                "exact_matches": SearchResults(
                                    results=results,
                                    query=query,
                                    search_method="uppercase_exact",
                                    successful=True
                                )
                            }
                    except Exception as e:
                        logger.error(f"Error parsing uppercase search results: {str(e)}")
            
            # If we get here, no results were found or there was an error
            return {
                "exact_matches": SearchResults(
                    results=[],
                    query=query,
                    search_method="uppercase_exact",
                    successful=False
                )
            }
            
        except Exception as e:
            logger.error(f"Error in uppercase exact search: {str(e)}")
            return {
                "exact_matches": SearchResults(
                    results=[],
                    query=query,
                    search_method="uppercase_exact",
                    successful=False
                ),
                "errors": state.get("errors", []) + [f"Error in uppercase search: {str(e)}"]
            }
    
    async def search_active_principle(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search by active principle, which is highly relevant for medication searches.
        """
        query = state["query"]
        session = state["session"]
        
        if not query.active_principle:
            # No active principle to search for, skip this step
            return {
                "active_principle_matches": SearchResults(
                    results=[],
                    query=query,
                    search_method="active_principle",
                    successful=True
                )
            }
        
        try:
            # Search by active principle
            active_principle = query.active_principle
            logger.info(f"Searching by active principle: {active_principle}")
            
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
                                            formulation_type=query.formulation_type
                                        )
                                        
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
                                            formulation_type=query.formulation_type
                                        )
                                        
                                        if result.relevance_score >= MIN_RELEVANCE_THRESHOLD:
                                            results.append(result)
                                            seen_nregistros.add(med.get("nregistro"))
                        except Exception as e:
                            logger.error(f"Error parsing practiv1 search results: {str(e)}")
            
            # Sort by relevance score
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            logger.info(f"Found {len(results)} active principle matches")
            
            return {
                "active_principle_matches": SearchResults(
                    results=results,
                    query=query,
                    search_method="active_principle",
                    successful=True
                )
            }
            
        except Exception as e:
            logger.error(f"Error in active principle search: {str(e)}")
            return {
                "active_principle_matches": SearchResults(
                    results=[],
                    query=query,
                    search_method="active_principle",
                    successful=False
                ),
                "errors": state.get("errors", []) + [f"Error in active principle search: {str(e)}"]
            }
    
    async def search_by_name(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search by complete name or query text.
        """
        query = state["query"]
        session = state["session"]
        
        try:
            # Search by complete query
            logger.info(f"Searching by name/query: {query.query_text}")
            
            search_url = f"{self.base_url}/medicamentos"
            results = []
            seen_nregistros = set()
            
            async with session.get(search_url, params={"nombre": query.query_text}) as response:
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
                                        formulation_type=query.formulation_type
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
            logger.info(f"Found {len(results)} name matches")
            
            return {
                "name_matches": SearchResults(
                    results=results,
                    query=query,
                    search_method="name",
                    successful=True
                )
            }
            
        except Exception as e:
            logger.error(f"Error in name search: {str(e)}")
            return {
                "name_matches": SearchResults(
                    results=[],
                    query=query,
                    search_method="name",
                    successful=False
                ),
                "errors": state.get("errors", []) + [f"Error in name search: {str(e)}"]
            }
    
    async def search_terms(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search by individual terms extracted from the query (last resort).
        """
        query = state["query"]
        session = state["session"]
        
        if not query.search_terms:
            # No search terms to use
            return {
                "fuzzy_matches": SearchResults(
                    results=[],
                    query=query,
                    search_method="terms",
                    successful=True
                )
            }
        
        try:
            logger.info(f"Searching by terms: {query.search_terms}")
            
            search_url = f"{self.base_url}/medicamentos"
            results = []
            seen_nregistros = set()
            
            # Only use the most promising search terms
            for term in query.search_terms[:3]:
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
                                            formulation_type=query.formulation_type
                                        )
                                        
                                        # Higher threshold for term searches to avoid irrelevant results
                                        if result.relevance_score >= MIN_RELEVANCE_THRESHOLD + 10:
                                            results.append(result)
                                            seen_nregistros.add(med.get("nregistro"))
                        except Exception as e:
                            logger.error(f"Error parsing term search results: {str(e)}")
            
            # Sort by relevance score
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            logger.info(f"Found {len(results)} term matches")
            
            return {
                "fuzzy_matches": SearchResults(
                    results=results,
                    query=query,
                    search_method="terms",
                    successful=True
                )
            }
            
        except Exception as e:
            logger.error(f"Error in term search: {str(e)}")
            return {
                "fuzzy_matches": SearchResults(
                    results=[],
                    query=query,
                    search_method="terms",
                    successful=False
                ),
                "errors": state.get("errors", []) + [f"Error in term search: {str(e)}"]
            }
    
    async def evaluate_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all search results and select the best ones.
        
        This node examines all results collected from different search strategies
        and determines which ones to return based on relevance and quality.
        """
        # Collect all results
        all_results: List[MedicationResult] = []
        seen_nregistros: Set[str] = set()
        result_quality = "no_results"
        
        # Helper to add unique results
        def add_unique_results(results: List[MedicationResult]) -> None:
            for result in results:
                if result.nregistro not in seen_nregistros:
                    all_results.append(result)
                    seen_nregistros.add(result.nregistro)
        
        # Add results from each search method in order of preference
        if state.get("exact_matches") and state["exact_matches"].results:
            add_unique_results(state["exact_matches"].results)
            result_quality = "high"
        
        if state.get("active_principle_matches") and state["active_principle_matches"].results:
            add_unique_results(state["active_principle_matches"].results)
            if result_quality == "no_results":
                result_quality = "medium"
        
        if state.get("name_matches") and state["name_matches"].results:
            add_unique_results(state["name_matches"].results)
            if result_quality == "no_results":
                result_quality = "low"
        
        if state.get("fuzzy_matches") and state["fuzzy_matches"].results:
            add_unique_results(state["fuzzy_matches"].results)
            if result_quality == "no_results":
                result_quality = "very_low"
        
        # Sort all results by relevance score
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Filter out results with low relevance scores
        final_results = [r for r in all_results if r.relevance_score >= MIN_RELEVANCE_THRESHOLD]
        
        # Limit to maximum results
        final_results = final_results[:MAX_RESULTS]
        
        # Check if we found anything
        if not final_results:
            result_quality = "no_results"
        
        logger.info(f"Final evaluation: {len(final_results)} results with quality {result_quality}")
        
        # Clean up the aiohttp session if we created one
        if state.get("session"):
            try:
                await state["session"].close()
            except Exception as e:
                logger.error(f"Error closing session: {str(e)}")
        
        return {
            "final_results": final_results,
            "result_quality": result_quality
        }
    
    def check_uppercase_results(self, state: Dict[str, Any]) -> str:
        """Determine next step based on uppercase search results."""
        if (state.get("exact_matches") and 
            state["exact_matches"].results and 
            len(state["exact_matches"].results) > 0):
            return "found_exact"
        return "not_found"
    
    def check_ap_results(self, state: Dict[str, Any]) -> str:
        """Determine next step based on active principle search results."""
        if (state.get("active_principle_matches") and 
            state["active_principle_matches"].results and 
            len(state["active_principle_matches"].results) > 0):
            return "found"
        return "not_found"
    
    def check_name_results(self, state: Dict[str, Any]) -> str:
        """Determine next step based on name search results."""
        if (state.get("name_matches") and 
            state["name_matches"].results and 
            len(state["name_matches"].results) > 0):
            return "found"
        return "not_found"
    
    def _calculate_relevance(self, 
                            med: MedicationResult, 
                            active_principle: Optional[str] = None,
                            concentration: Optional[str] = None,
                            query_terms: Optional[List[str]] = None,
                            formulation_type: Optional[str] = None) -> int:
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
            if active_principle.lower() in pactivos_lower:
                # Full match in active principles
                score += 100
            elif active_principle.lower() in med_name_lower:
                # Active principle appears in name
                score += 50
        
        # Check for concentration match
        if concentration and concentration in med_name_lower:
            score += 30
        
        # Check formulation type match
        if formulation_type and formulation_type in med_name_lower:
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
    
    async def execute_search(self, query_text: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Execute the complete search workflow.
        
        Args:
            query_text: The search query
            
        Returns:
            Tuple[List[Dict], str]: List of results and quality assessment
        """
        # Prepare initial state
        query = MedicationQuery(query_text=query_text)
        initial_state = {"query": query}
        
        # Execute the workflow
        try:
            # LangGraph execution
            final_state = await self.workflow.acall(initial_state)
            
            # Extract and return results
            final_results = final_state.get("final_results", [])
            result_quality = final_state.get("result_quality", "unknown")
            
            # Convert Pydantic models to dictionaries for easier integration
            return [result.dict() for result in final_results], result_quality
            
        except Exception as e:
            logger.error(f"Error executing search workflow: {str(e)}")
            return [], "error"