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
Eres un experto farmacéutico especializado en medicamentos registrados en CIMA (Centro de Información online de Medicamentos de la AEMPS).

Tu objetivo es proporcionar información precisa y detallada sobre medicamentos en respuesta a consultas de usuarios.

Cuando respondas a la siguiente consulta:
1. Primero, muestra tu **proceso de razonamiento** paso a paso de forma explícita
2. Luego proporciona una **respuesta completa** basada en tu razonamiento
3. Finalmente, incluye una sección de **referencias** con al menos 3-5 fuentes específicas utilizadas

Utiliza el siguiente formato:

## PROCESO DE RAZONAMIENTO
- Paso 1: [Un análisis inicial de la consulta]
- Paso 2: [Considerar las posibles interpretaciones o aspectos relevantes]
- Paso 3: [Evaluar la información más importante a incluir]
- Paso 4: [Identificar posibles precauciones o advertencias relevantes]
- Paso 5: [Determinar conclusiones y recomendaciones]

## RESPUESTA
[Tu respuesta completa y detallada]

## REFERENCIAS
1. [Título del documento o fuente] - [URL o información de identificación]
2. [Título del documento o fuente] - [URL o información de identificación]
3. [Título del documento o fuente] - [URL o información de identificación]

La consulta es: "{question}"
        """.strip()

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
            logger.info(f"Sending request to Perplexity API for question: {question}")
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            result = response.json()
            
            # Extract and process the response
            if result.get("choices") and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                
                # Extract reasoning and references from the structured response
                reasoning = ""
                answer = content
                references = []
                
                # Extract reasoning section
                reasoning_section = self._extract_section(content, "PROCESO DE RAZONAMIENTO", "RESPUESTA")
                if reasoning_section:
                    reasoning = reasoning_section
                
                # Extract answer section
                answer_section = self._extract_section(content, "RESPUESTA", "REFERENCIAS")
                if answer_section:
                    answer = answer_section
                
                # Extract references section
                references_section = self._extract_section(content, "REFERENCIAS", None)
                if references_section:
                    # Process references into a list
                    references = self._extract_references(references_section)
                
                # Update conversation history for context in future questions
                self.conversation_history.append({"role": "user", "content": question})
                self.conversation_history.append({"role": "assistant", "content": content})
                
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
    
    def _extract_section(self, content: str, start_marker: str, end_marker: Optional[str]) -> str:
        """Extract a section from the content between start_marker and end_marker"""
        try:
            start_index = content.find(f"## {start_marker}")
            if start_index == -1:
                return ""
            
            start_index = start_index + len(f"## {start_marker}")
            
            if end_marker:
                end_index = content.find(f"## {end_marker}", start_index)
                if end_index == -1:
                    section = content[start_index:].strip()
                else:
                    section = content[start_index:end_index].strip()
            else:
                section = content[start_index:].strip()
            
            return section
        except Exception as e:
            logger.error(f"Error extracting section {start_marker}: {str(e)}")
            return ""
    
    def _extract_references(self, references_section: str) -> List[Dict[str, str]]:
        """Extract references from the references section into a structured format"""
        references = []
        lines = references_section.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove leading numbers or bullets
            line = line.lstrip("0123456789.-*[] \t")
            
            # Try to split into title and URL if possible
            parts = line.split(" - ", 1)
            if len(parts) == 2:
                title, url = parts
                references.append({"title": title.strip(), "url": url.strip()})
            else:
                references.append({"title": line, "url": ""})
        
        return references
    
    async def ask_cima_question_async(self, question: str) -> Dict[str, Any]:
        """Async wrapper around the synchronous API call"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.ask_cima_question, question)
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
