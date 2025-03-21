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
        # Construct a simplified prompt specifically requesting references
        prompt = f"""
Eres un asistente farmacéutico especializado en información sobre medicamentos.

CONSULTA ACTUAL: "{question}"

Responde a esta consulta sobre medicamentos siguiendo estas instrucciones:

1. Analiza la consulta de manera lógica
2. Proporciona información precisa y específica sobre medicamentos
3. Es FUNDAMENTAL que incluyas TODAS las referencias utilizadas en tu respuesta, citando fuentes académicas, publicaciones médicas, o bases de datos farmacéuticas

Estructura tu respuesta en tres partes claramente separadas:

PARTE 1: ANÁLISIS
[Aquí explica tu razonamiento, considerando los aspectos médicos relevantes]

PARTE 2: RESPUESTA
[Aquí proporciona la respuesta completa a la consulta]

PARTE 3: REFERENCIAS
[IMPORTANTE: Enumera TODAS las fuentes consultadas, incluyendo links completos cuando sea posible. Incluir al menos 5-7 referencias específicas.]
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
                
                # If we couldn't extract sections properly, fallback to searching for them in the content
                if not answer:
                    # Try a more aggressive approach to find the answer section
                    for header in possible_answer_headers:
                        pattern = re.compile(f"(?:^|\n)(?:#{1,3}\\s*|\\*\\*)?{header}\\b", re.IGNORECASE)
                        if pattern.search(content):
                            # Found a header, get everything after it
                            parts = pattern.split(content, 1)
                            if len(parts) > 1:
                                answer = parts[1].strip()
                                # Try to end the answer at the next section if possible
                                for ref_header in possible_reference_headers:
                                    ref_pattern = re.compile(f"(?:^|\n)(?:#{1,3}\\s*|\\*\\*)?{ref_header}\\b", re.IGNORECASE)
                                    if ref_pattern.search(answer):
                                        answer_parts = ref_pattern.split(answer, 1)
                                        if len(answer_parts) > 1:
                                            answer = answer_parts[0].strip()
                                            # Save the references part
                                            references_text = answer_parts[1].strip()
                                            break
                                break
                
                # Try to find references even if we couldn't identify the references section
                if not references_text:
                    # Look for URLs anywhere in the content as a last resort
                    all_urls = re.findall(r'https?://[^\s\)\]]+', content)
                    if all_urls:
                        references_text = "URLs encontradas en el texto:\n" + "\n".join(all_urls)
                
                # If we still don't have an answer, use the whole content
                if not answer:
                    logger.warning("Could not extract structured sections - using full content")
                    answer = content
                
                # Always try to find references in the text, even if the references section wasn't found
                references = self._extract_all_references(content, references_text)
                
                # Extract URLs directly from the text for backup references
                urls_in_text = re.findall(r'https?://[^\s\)\]]+', content)
                for url in urls_in_text:
                    # Add as reference if not already included
                    if url and not any(url in ref.get("url", "") for ref in references):
                        domain = re.search(r'https?://(?:www\.)?([^/]+)', url)
                        title = domain.group(1) if domain else "Fuente online"
                        references.append({"title": title, "url": url})
                
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
    
    def _extract_all_references(self, full_content: str, references_text: str) -> List[Dict[str, str]]:
        """
        Extract ALL references from both the references section and full content.
        This is an enhanced method that searches for references throughout the text.
        
        Args:
            full_content: The complete response content
            references_text: The specific references section if found
            
        Returns:
            List of all found references
        """
        all_references = []
        
        # First try to extract from dedicated references section if available
        if references_text:
            references_from_section = self._extract_references_from_section(references_text)
            all_references.extend(references_from_section)
        
        # Then look for references throughout the full content
        # Extract URLs
        all_urls = re.findall(r'https?://[^\s\(\)\[\]\"\']+', full_content)
        
        # Look for citation-like patterns in the text
        citation_patterns = [
            # Numbered citations [1], [2], etc.
            r'\[\d+\]\s*([^[]+?)(?=\[\d+\]|\n\n|$)',
            # Parenthetical citations (Author, year)
            r'\([A-Z][a-zA-Z]+(?:\set\sal\.?|\sy\s[A-Z][a-zA-Z]+)?,\s*\d{4}\)',
            # Standard reference formats
            r'[A-Z][a-zA-Z\-]+,\s[A-Z]\.\s*(?:et\sal\.)?\s\(\d{4}\)\.'
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, full_content)
            for match in matches:
                # Check if this could be a reference
                if len(match) > 15 and not any(match in ref.get("title", "") for ref in all_references):
                    # Look for URLs in this match
                    url_in_match = re.search(r'(https?://[^\s\(\)\[\]\"\']+)', match)
                    url = url_in_match.group(1) if url_in_match else ""
                    all_references.append({"title": match.strip(), "url": url})
        
        # Look for more structured references in the full content
        # 1. Try to find numbered references
        numbered_refs = re.findall(r'(?:^|\n)\s*\d+\.?\s+([^\n]+)', full_content, re.MULTILINE)
        for ref in numbered_refs:
            if len(ref) > 20 and not any(ref in r.get("title", "") for r in all_references):
                # This looks like a reference
                url_in_ref = re.search(r'(https?://[^\s\(\)\[\]\"\']+)', ref)
                url = url_in_ref.group(1) if url_in_ref else ""
                
                # Clean up the reference
                clean_ref = ref.strip()
                if url:
                    clean_ref = clean_ref.replace(url, "").strip()
                
                all_references.append({"title": clean_ref, "url": url})
        
        # 2. Look for bulleted references
        bulleted_refs = re.findall(r'(?:^|\n)\s*[\-\*•]\s+([^\n]+)', full_content, re.MULTILINE)
        for ref in bulleted_refs:
            if len(ref) > 20 and not any(ref in r.get("title", "") for r in all_references):
                # This looks like a reference
                url_in_ref = re.search(r'(https?://[^\s\(\)\[\]\"\']+)', ref)
                url = url_in_ref.group(1) if url_in_ref else ""
                
                # Clean up the reference
                clean_ref = ref.strip()
                if url:
                    clean_ref = clean_ref.replace(url, "").strip()
                
                all_references.append({"title": clean_ref, "url": url})
        
        # Add any URLs that weren't already included
        for url in all_urls:
            if not any(url == r.get("url", "") for r in all_references):
                # Extract domain for title
                domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
                title = f"Fuente: {domain_match.group(1)}" if domain_match else "Fuente online"
                all_references.append({"title": title, "url": url})
        
        # If no references were found, add some standard medical sources
        if not all_references:
            all_references = [
                {"title": "Base de datos CIMA (AEMPS)", "url": "https://cima.aemps.es/"},
                {"title": "Vademecum - Información de medicamentos", "url": "https://www.vademecum.es/"},
                {"title": "Agencia Española de Medicamentos y Productos Sanitarios", "url": "https://www.aemps.gob.es/"},
                {"title": "PubMed - Biblioteca Nacional de Medicina de EE.UU.", "url": "https://pubmed.ncbi.nlm.nih.gov/"},
                {"title": "DrugBank - Base de datos de medicamentos", "url": "https://go.drugbank.com/"},
                {"title": "Medscape - Referencia médica", "url": "https://reference.medscape.com/"}
            ]
        
        # Remove duplicates while preserving order
        unique_references = []
        seen_titles = set()
        seen_urls = set()
        
        for ref in all_references:
            title = ref.get("title", "")
            url = ref.get("url", "")
            
            # Skip if we've seen this exact title or URL before
            if (title and title in seen_titles) or (url and url in seen_urls):
                continue
                
            if title:
                seen_titles.add(title)
            if url:
                seen_urls.add(url)
                
            unique_references.append(ref)
        
        return unique_references
        
    def _extract_references_from_section(self, references_text: str) -> List[Dict[str, str]]:
        """
        Extract references specifically from the references section
        
        Args:
            references_text: The text containing the references
            
        Returns:
            List of extracted references
        """
        if not references_text:
            return []
            
        references = []
        
        # Try multiple extraction methods to get all possible references
        
        # Method 1: Look for numbered references
        numbered_refs = re.findall(r'(?:^|\n)\s*\d+\.?\s+([^\n]+)', references_text)
        for ref in numbered_refs:
            if len(ref) > 10:  # Ignore very short lines
                # Try to extract URL
                url_match = re.search(r'(https?://[^\s\)\]\"\'\:]+)', ref)
                url = url_match.group(1) if url_match else ""
                
                # Clean up title
                title = ref
                if url:
                    title = title.replace(url, "").strip()
                
                # Clean up formatting
                title = re.sub(r'^\s*\d+[\.\)]\s*', '', title)  # Remove leading numbers
                title = title.strip(' "\'.,:-')
                
                references.append({"title": title, "url": url})
        
        # Method 2: Look for bulleted references
        if not references:
            bulleted_refs = re.findall(r'(?:^|\n)\s*[\-\*•]\s+([^\n]+)', references_text)
            for ref in bulleted_refs:
                if len(ref) > 10:  # Ignore very short lines
                    # Try to extract URL
                    url_match = re.search(r'(https?://[^\s\)\]\"\'\:]+)', ref)
                    url = url_match.group(1) if url_match else ""
                    
                    # Clean up title
                    title = ref
                    if url:
                        title = title.replace(url, "").strip()
                    
                    # Clean up formatting
                    title = re.sub(r'^\s*[\-\*•]\s*', '', title)  # Remove bullet indicators
                    title = title.strip(' "\'.,:-')
                    
                    references.append({"title": title, "url": url})
        
        # Method 3: Split by new lines if no structured references were found
        if not references:
            lines = references_text.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 15:  # Skip short lines
                    # Try to extract URL
                    url_match = re.search(r'(https?://[^\s\)\]\"\'\:]+)', line)
                    url = url_match.group(1) if url_match else ""
                    
                    # Clean up title
                    title = line
                    if url:
                        title = title.replace(url, "").strip()
                    else:
                        # See if there's a URL marker
                        url_indicator = re.search(r'Disponible en:\s*(\S+)', line, re.IGNORECASE)
                        if url_indicator:
                            potential_url = url_indicator.group(1)
                            if re.match(r'https?://', potential_url):
                                url = potential_url
                    
                    # Clean up title more
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
