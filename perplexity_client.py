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
            dict: Structured response with answer and metadata
        """
        # Construct a prompt instructing the API to output structured information
        prompt = f"""
Eres un experto farmacéutico especializado en medicamentos registrados en CIMA (Centro de Información online de Medicamentos de la AEMPS).

Tu objetivo es proporcionar información precisa y detallada sobre medicamentos en respuesta a consultas de usuarios.

La consulta es: "{question}"

Proporciona una respuesta completa, estructurada y detallada. Si la consulta menciona un medicamento específico, incluye:
- Información sobre el principio activo
- Indicaciones terapéuticas
- Contraindicaciones relevantes
- Posología y forma de administración (si aplica)
- Efectos adversos importantes
- Precauciones especiales

En cambio, si la consulta es sobre un tema general o una clase terapéutica, proporciona una visión general relevante y útil.

Estructura tu respuesta de manera clara utilizando encabezados y listas cuando sea apropiado. Incluye referencias a fuentes oficiales en línea de la AEMPS si conoces alguna URL específica.
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
            
            # Extract and return the response
            if result.get("choices") and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                
                # Update conversation history for context in future questions
                self.conversation_history.append({"role": "user", "content": question})
                self.conversation_history.append({"role": "assistant", "content": content})
                
                # Limit conversation history to last 10 messages (5 exchanges)
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
                
                return {
                    "answer": content,
                    "context": "Generado con Perplexity Sonar API",
                    "success": True
                }
            else:
                logger.error("No answer provided in Perplexity API response")
                return {
                    "answer": "Lo siento, no he podido generar una respuesta. Por favor, intenta reformular tu pregunta.",
                    "context": "Error: No se recibió respuesta de la API",
                    "success": False
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return {
                "answer": f"Error de conexión con el servicio: {str(e)}",
                "context": "Error de conexión",
                "success": False
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return {
                "answer": "Error al procesar la respuesta del servicio.",
                "context": "Error de formato",
                "success": False
            }
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {
                "answer": "Ha ocurrido un error inesperado. Por favor, inténtalo de nuevo más tarde.",
                "context": f"Error: {str(e)}",
                "success": False
            }
    
    async def ask_cima_question_async(self, question: str) -> Dict[str, Any]:
        """Async wrapper around the synchronous API call"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.ask_cima_question, question)
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
