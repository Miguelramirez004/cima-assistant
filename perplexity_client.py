"""Perplexity Sonar API Client for CIMA Assistant"""
import requests
import json
import logging
import asyncio
import os
from typing import Dict, Any, List, Optional
import backoff

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerplexityClient:
    """Client for interacting with Perplexity's Sonar API"""
    
    def __init__(self, api_key: str):
        """Initialize the Perplexity API client"""
        self.api_key = api_key
        self.base_url = 'https://api.perplexity.ai/chat/completions'
        self.model = 'sonar-pro'
        self.conversation_history = []
    
    @backoff.on_exception(backoff.expo, 
                          (requests.exceptions.RequestException,
                           requests.exceptions.ConnectionError,
                           requests.exceptions.Timeout), 
                          max_tries=3)
    def ask_cima_question(self, question: str) -> Dict[str, Any]:
        """
        Send a question about medications to Perplexity API and parse the response.
        
        Args:
            question (str): The question about a medication or pharmaceutical topic
            
        Returns:
            dict: Structured response with answer, reasoning steps, and citations
        """
        # Construct a prompt instructing the API to output structured information with reasoning steps and references
        prompt = f"""
Eres un experto farmacéutico especializado en medicamentos.

Tu tarea es responder a consultas sobre medicamentos, sus efectos, usos, contraindicaciones, etc.
Para esta consulta específica, debes:

1. Realizar un análisis detallado paso a paso
2. Proporcionar una respuesta completa
3. Incluir referencias a fuentes médicas confiables

Estructura tu respuesta con exactamente estas secciones:

## PROCESO DE RAZONAMIENTO
[Aquí desarrolla paso a paso tu análisis del problema]

## RESPUESTA
[Aquí proporciona la respuesta completa]

## REFERENCIAS
[Incluye al menos 3 referencias a fuentes médicas]

La consulta es: "{question}"
        """.strip()

        # Log that we're making a Perplexity API request
        logger.info(f"Sending request to Perplexity API with query: '{question}'")
        
        # Build the payload
        payload = {
            "model": self.model,
            "messages": self.conversation_history + [
                {"role": "user", "content": prompt}
            ]
        }

        # Set up headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # Make the API request
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            result = response.json()
            
            # Extract and process the response
            if result.get("choices") and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                
                # Log the raw response for debugging
                logger.info(f"Raw Perplexity response: {content[:200]}...")
                
                # Extract reasoning and references from the structured response
                reasoning_section = self._extract_section(content, "PROCESO DE RAZONAMIENTO", "RESPUESTA")
                answer_section = self._extract_section(content, "RESPUESTA", "REFERENCIAS") 
                references_section = self._extract_section(content, "REFERENCIAS", None)
                
                # If sections aren't found, try alternative methods of extraction
                if not reasoning_section and not answer_section:
                    # Try different heading patterns
                    reasoning_section = self._extract_flexible_section(content, 
                                                                     ["PROCESO DE RAZONAMIENTO", "RAZONAMIENTO", "ANÁLISIS", "PENSAMIENTO"])
                    answer_section = self._extract_flexible_section(content, 
                                                                  ["RESPUESTA", "CONTESTACIÓN", "INFORMACIÓN", "RESULTADO"])
                    references_section = self._extract_flexible_section(content, 
                                                                      ["REFERENCIAS", "FUENTES", "BIBLIOGRAFÍA", "CITAS"])
                
                # If we still can't extract sections, just use the whole content as the answer
                if not answer_section:
                    logger.warning("Could not extract structured sections - using full content as answer")
                    answer_section = content
                
                # Process references into a list
                references = self._extract_references(references_section) if references_section else []
                
                # Update conversation history for context in future questions
                self.conversation_history.append({"role": "user", "content": question})
                self.conversation_history.append({"role": "assistant", "content": content})
                
                # Limit conversation history to last 10 messages (5 exchanges)
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
                
                return {
                    "answer": answer_section,
                    "reasoning": reasoning_section,
                    "references": references,
                    "full_content": content,
                    "context": "Generado con Perplexity Sonar API",
                    "success": True
                }
            else:
                logger.error("No answer provided in Perplexity API response")
                return {
                    "answer": "Lo siento, no he podido generar una respuesta. Por favor, intenta reformular tu pregunta.",
                    "reasoning": "",
                    "references": [],
                    "context": "Error: No se recibió respuesta de la API",
                    "success": False
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return {
                "answer": f"Error de conexión con el servicio: {str(e)}",
                "reasoning": "",
                "references": [],
                "context": "Error de conexión",
                "success": False
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return {
                "answer": "Error al procesar la respuesta del servicio.",
                "reasoning": "",
                "references": [],
                "context": "Error de formato",
                "success": False
            }
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {
                "answer": "Ha ocurrido un error inesperado. Por favor, inténtalo de nuevo más tarde.",
                "reasoning": "",
                "references": [],
                "context": f"Error: {str(e)}",
                "success": False
            }
    
    def _extract_section(self, content: str, start_marker: str, end_marker: Optional[str]) -> str:
        """Extract a section from the content between start_marker and end_marker"""
        try:
            # Try multiple variations of the heading marker
            heading_patterns = [
                f"## {start_marker}",
                f"# {start_marker}",
                f"**{start_marker}**",
                f"**{start_marker}:**",
                f"{start_marker}:"
            ]
            
            # Try each pattern
            for pattern in heading_patterns:
                start_index = content.find(pattern)
                if start_index != -1:
                    # Found a match
                    start_index = start_index + len(pattern)
                    
                    # Search for end marker if provided
                    if end_marker:
                        # Try various formats for end marker
                        end_patterns = [
                            f"## {end_marker}",
                            f"# {end_marker}",
                            f"**{end_marker}**",
                            f"**{end_marker}:**",
                            f"{end_marker}:"
                        ]
                        
                        # Try each end pattern
                        found_end = False
                        for end_pattern in end_patterns:
                            end_index = content.find(end_pattern, start_index)
                            if end_index != -1:
                                section = content[start_index:end_index].strip()
                                found_end = True
                                break
                        
                        if not found_end:
                            # If no end pattern found, take everything to the end
                            section = content[start_index:].strip()
                    else:
                        # No end marker, take everything to the end
                        section = content[start_index:].strip()
                    
                    # Clean up the section
                    section = section.strip()
                    return section
            
            # No match found for any pattern
            logger.warning(f"Section not found: {start_marker}")
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting section {start_marker}: {str(e)}")
            return ""
            
    def _extract_flexible_section(self, content: str, possible_markers: List[str]) -> str:
        """Try to extract a section using multiple possible section titles"""
        for marker in possible_markers:
            # Try with various end markers
            for end_marker in ["RESPUESTA", "REFERENCIAS", "CONCLUSIÓN", "CONCLUSION", None]:
                if marker == end_marker:
                    continue
                section = self._extract_section(content, marker, end_marker)
                if section:
                    return section
        return ""
    
    def _extract_references(self, references_section: str) -> List[Dict[str, str]]:
        """Extract references from the references section into a structured format"""
        if not references_section:
            return []
            
        references = []
        # Split by numbered list patterns
        potential_refs = re.split(r'\n\s*\d+[\.\)]\s*', references_section)
        
        # If that didn't work well, try splitting by new lines
        if len(potential_refs) <= 1:
            potential_refs = references_section.split('\n')
        
        for ref in potential_refs:
            ref = ref.strip()
            if not ref or ref.isdigit() or len(ref) < 10:  # Skip empty or very short lines
                continue
                
            # Try to extract URL if present
            url_match = re.search(r'https?://[^\s\)"\']+', ref)
            if url_match:
                url = url_match.group(0)
                title = ref.replace(url, '').strip()
                # Clean up title
                title = title.strip(' "\'.,:-')
                if not title:
                    title = url
            else:
                url = ""
                title = ref
            
            # Clean up title
            title = re.sub(r'^\s*\d+[\.\)]\s*', '', title)  # Remove leading numbers
            title = title.strip(' "\'.,:-')
            
            if title:
                references.append({"title": title, "url": url})
        
        return references
    
    async def ask_cima_question_async(self, question: str) -> Dict[str, Any]:
        """Async wrapper around the synchronous API call"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.ask_cima_question, question)
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
