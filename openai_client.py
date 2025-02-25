"""
Custom OpenAI client creation to handle compatibility issues
"""
from openai import AsyncOpenAI, OpenAI

def create_async_openai_client(api_key):
    """
    Create an AsyncOpenAI client with limited configuration to avoid
    proxy-related errors on Streamlit Cloud.
    """
    # Create with minimal settings to avoid any proxy issues
    client = AsyncOpenAI(
        api_key=api_key
    )
    return client

def create_openai_client(api_key):
    """
    Create a regular OpenAI client as a fallback if needed
    """
    client = OpenAI(
        api_key=api_key
    )
    return client