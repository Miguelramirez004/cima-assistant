import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI Model Configuration for v0.27.8
    CHAT_MODEL = "gpt-4o-mini"  # Compatible with older OpenAI SDK
    EMBEDDING_MODEL = "text-embedding-ada-002"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Perplexity API Configuration
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    PERPLEXITY_MODEL = "sonar-reasoning"  # Sonar model with reasoning capabilities
    
    # CIMA API Configuration
    CIMA_BASE_URL = "https://cima.aemps.es/cima/rest"
    
    # Application Settings
    MAX_RESULTS = 5  # Maximum number of medication results to retrieve
    CACHE_ENABLED = True  # Enable/disable caching
    CACHE_TIMEOUT = 3600  # Cache timeout in seconds (1 hour)
    
    # Feature Flags
    USE_PERPLEXITY = True  # Whether to use Perplexity for CIMA consultations
    
    # RAG Settings
    CONTEXT_SIZE = 8000  # Maximum context size in tokens
    
    # Formulation Settings
    FORMULATION_TYPES = {
        "suspension": ["suspension", "suspensión"],
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
    
    # Administration Routes
    ADMIN_ROUTES = {
        "oral": ["oral", "vía oral", "via oral", "por boca"],
        "topica": ["tópica", "topica", "cutánea", "cutanea"],
        "oftalmico": ["oftálmico", "oftalmico", "ocular"],
        "rectal": ["rectal", "vía rectal", "via rectal"],
        "nasal": ["nasal", "intranasal"],
        "otico": ["ótico", "otico", "auricular"],
        "vaginal": ["vaginal", "intravaginal"],
        "parenteral": ["parenteral", "inyectable", "inyección", "inyeccion"]
    }
