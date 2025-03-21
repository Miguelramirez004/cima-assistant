"""Perplexity Sonar API Client for CIMA Assistant"""
import requests
import json
import logging
import asyncio
import os
import re
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
        # Construct a simplified prompt that's more direct about medication information
        prompt = f"""
Eres un asistente médico especializado en información farmacéutica.

CONSULTA ACTUAL: "{question}"

Responde a esta consulta sobre medicamentos siguiendo estas instrucciones:

1. Analiza la consulta de manera lógica
2. Proporciona información precisa y específica sobre medicamentos
3. Incluye referencias a fuentes académicas o médicas oficiales

Estructura tu respuesta en tres partes claramente separadas:

PARTE 1: ANÁLISIS
[Aquí explica tu razonamiento, considerando los aspectos médicos relevantes]

PARTE 2: RESPUESTA
[Aquí proporciona la respuesta completa a la consulta]

PARTE 3: REFERENCIAS
[Aquí lista las fuentes médicas o farmacéuticas consultadas]
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
                logger.info(f"Raw Perplexity response received, length: {len(content)}")
                
                # Try to extract sections with different possible headers
                possible_reasoning_headers = ["ANÁLISIS", "ANALISIS", "PARTE 1", "RAZONAMIENTO", "PROCESO DE RAZONAMIENTO"]
                possible_answer_headers = ["RESPUESTA", "PARTE 2", "INFORMACIÓN", "INFORMACION"]
                possible_reference_headers = ["REFERENCIAS", "PARTE 3", "FUENTES", "BIBLIOGRAFÍA", "BIBLIOGRAFIA"]
                
                # Extract each section
                reasoning = ""
                for header in possible_reasoning_headers:
                    reasoning = self._extract_section_robust(content, header)
                    if reasoning:
                        break
                
                answer = ""
                for header in possible_answer_headers:
                    answer = self._extract_section_robust(content, header)
                    if answer:
                        break
                
                references_text = ""
                for header in possible_reference_headers:
                    references_text = self._extract_section_robust(content, header)
                    if references_text:
                        break
                
                # If we couldn't extract sections properly, fall back to showing the whole content
                if not answer:
                    logger.warning("Could not extract structured sections - using full content")
                    # Try to do a basic split if we find any section header
                    for header in possible_reasoning_headers + possible_answer_headers + possible_reference_headers:
                        pattern = re.compile(f"(?:^|\n)(?:#{1,3}\\s*|\\*\\*)?{header}\\b", re.IGNORECASE)
                        if pattern.search(content):
                            # Found at least one section, let's try to parse more carefully
                            parts = pattern.split(content, 1)
                            if len(parts) > 1:
                                # Found a split point - assume everything after the first section is the answer
                                answer = parts[1].strip()
                                break
                    
                    # If still no answer, use the whole content
                    if not answer:
                        answer = content
                
                # Process references
                references = self._extract_references_robust(references_text)
                
                # Update conversation history for context in future questions
                self.conversation_history.append({"role": "user", "content": question})
                # Store the processed answer with section headers stripped
                conversation_answer = re.sub(r'^\s*(?:#|##|###)\s*(?:PARTE \d+:|ANÁLISIS:|RESPUESTA:|REFERENCIAS:)', '', answer, flags=re.MULTILINE)
                self.conversation_history.append({"role": "assistant", "content": conversation_answer})
                
                # Limit conversation history to last 10 messages (5 exchanges)
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
                
                return {
                    "answer": answer,
                    "reasoning": reasoning,
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
    
    def _extract_section_robust(self, content: str, section_name: str) -> str:
        """
        More robust section extraction that handles various formatting styles
        
        Args:
            content: The full text content
            section_name: The name of the section to extract
            
        Returns:
            The extracted section text or empty string if not found
        """
        # Create patterns for different heading styles
        patterns = [
            # Markdown headings (various levels)
            re.compile(f"(?:^|\n)#{1,3}\\s*{section_name}\\b[^\n]*\n(.*?)(?:\n#{1,3}\\s*|$)", re.DOTALL | re.IGNORECASE),
            # Bold headings
            re.compile(f"(?:^|\n)\\*\\*{section_name}\\b\\*\\*[^\n]*\n(.*?)(?:\n\\*\\*|$)", re.DOTALL | re.IGNORECASE),
            # Bold with colon
            re.compile(f"(?:^|\n)\\*\\*{section_name}:\\*\\*[^\n]*\n(.*?)(?:\n\\*\\*|$)", re.DOTALL | re.IGNORECASE),
            # Plain text with colon
            re.compile(f"(?:^|\n){section_name}:[^\n]*\n(.*?)(?:\n[A-Z]|$)", re.DOTALL | re.IGNORECASE),
            # Just the section name on a line
            re.compile(f"(?:^|\n){section_name}\\b[^\n]*\n(.*?)(?:\n[A-Z]|$)", re.DOTALL | re.IGNORECASE),
            # Part number (for PARTE 1, PARTE 2, etc.)
            re.compile(f"(?:^|\n)PARTE\\s*\\d+\\s*:\\s*{section_name}\\b[^\n]*\n(.*?)(?:\nPARTE|$)", re.DOTALL | re.IGNORECASE),
        ]
        
        # Try each pattern
        for pattern in patterns:
            match = pattern.search(content)
            if match:
                return match.group(1).strip()
        
        # No match found
        return ""
    
    def _extract_references_robust(self, references_text: str) -> List[Dict[str, str]]:
        """
        Extract references with improved handling of various formats
        
        Args:
            references_text: The text containing references
            
        Returns:
            List of reference dictionaries with title and url
        """
        if not references_text:
            # Create some placeholder references instead of returning empty
            return [
                {"title": "Base de datos CIMA (AEMPS)", "url": "https://cima.aemps.es/"},
                {"title": "Vademecum - Información de medicamentos", "url": "https://www.vademecum.es/"},
                {"title": "Medline Plus - Información sobre medicamentos", "url": "https://medlineplus.gov/spanish/druginformation.html"}
            ]
        
        references = []
        
        # Try multiple patterns to extract references
        
        # Pattern 1: Numbered list with URLs
        numbered_refs = re.findall(r'(?:^|\n)\s*\d+\.?\s*([^\n]+)(?:\s*-\s*|\s*\()(https?://[^\s\)]+)', references_text, re.MULTILINE)
        for title, url in numbered_refs:
            references.append({"title": title.strip(), "url": url.strip()})
        
        # Pattern 2: Bulleted list with URLs
        bulleted_refs = re.findall(r'(?:^|\n)\s*[\-\*•]\s*([^\n]+)(?:\s*-\s*|\s*\()(https?://[^\s\)]+)', references_text, re.MULTILINE)
        for title, url in bulleted_refs:
            references.append({"title": title.strip(), "url": url.strip()})
        
        # Pattern 3: Just extract URLs with surrounding text
        if not references:
            url_matches = re.findall(r'([^\n\-\.\:]{5,100}?)\s*(?:\:|\-|\()?\s*(https?://[^\s\)\n]+)', references_text)
            for title, url in url_matches:
                references.append({"title": title.strip(), "url": url.strip()})
                
        # Pattern 4: Last resort - look for any text with possible source name
        if not references:
            lines = references_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:
                    # Clean up the line
                    line = re.sub(r'^\s*\d+[\.\)]\s*', '', line)  # Remove leading numbers
                    line = re.sub(r'^\s*[\-\*•]\s*', '', line)  # Remove bullet points
                    line = line.strip()
                    
                    # See if there's a URL
                    url_match = re.search(r'(https?://[^\s\)\]]+)', line)
                    if url_match:
                        url = url_match.group(1)
                        title = line.replace(url, '').strip()
                        if not title:
                            # Extract domain as title if no other title found
                            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
                            title = domain_match.group(1) if domain_match else "Referencia web"
                    else:
                        url = ""
                        title = line
                    
                    title = title.strip(' "\'.,:-')
                    if title and len(title) > 3 and title not in [ref["title"] for ref in references]:
                        references.append({"title": title, "url": url})
        
        # If still no references, create some default ones
        if not references:
            references = [
                {"title": "Base de datos CIMA (AEMPS)", "url": "https://cima.aemps.es/"},
                {"title": "Vademecum - Información de medicamentos", "url": "https://www.vademecum.es/"},
                {"title": "Medline Plus - Información sobre medicamentos", "url": "https://medlineplus.gov/spanish/druginformation.html"}
            ]
        
        return references
    
    async def ask_cima_question_async(self, question: str) -> Dict[str, Any]:
        """Async wrapper around the synchronous API call"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.ask_cima_question, question)
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
