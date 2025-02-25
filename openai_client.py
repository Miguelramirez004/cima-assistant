"""
Custom OpenAI client creation to handle compatibility issues
"""
from openai import AsyncOpenAI, OpenAI
import httpx  # Add this import

def create_async_openai_client(api_key: str, proxy: str = None):
    """Create an async client with optional proxy support."""
    if proxy:
        # Configure proxy via httpx.AsyncClient
        http_client = httpx.AsyncClient(proxies=proxy)
    else:
        http_client = None  # Let OpenAI use its default client

    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        http_client=http_client  # Pass the custom client
    )

def create_openai_client(api_key: str, proxy: str = None):
    """Create a sync client with proxy support."""
    return OpenAI(
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        # For sync clients, configure proxies via environment variables
    )
