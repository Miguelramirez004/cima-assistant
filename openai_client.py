"""Custom OpenAI client creation with robust connection handling and error resilience"""
from openai import AsyncOpenAI, OpenAI
import httpx
import backoff
import logging
import tenacity
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def create_async_openai_client(api_key: str, proxy: str = None, timeout: int = 60):
    """
    Create an async client with improved error handling and connection resilience.
    
    Args:
        api_key: OpenAI API key
        proxy: Optional proxy URL
        timeout: Request timeout in seconds
        
    Returns:
        AsyncOpenAI client instance
    """
    # Set up httpx timeout settings
    timeout_settings = httpx.Timeout(
        timeout,  # Default timeout
        connect=10.0,  # Connection timeout
        read=timeout,  # Read timeout
        write=timeout,  # Write timeout
        pool=5.0  # Pool timeout
    )
    
    # Configure limits
    limits = httpx.Limits(
        max_keepalive_connections=5,
        max_connections=10,
        keepalive_expiry=30.0  # 30 seconds
    )
    
    # Create common client settings
    client_settings = {
        "timeout": timeout_settings,
        "limits": limits,
        "follow_redirects": True,
        "http2": True
    }
    
    # Add proxy if provided
    if proxy:
        client_settings["proxies"] = {
            "http://": proxy,
            "https://": proxy
        }
    
    # Create the httpx client with robust retry logic
    http_client = httpx.AsyncClient(**client_settings)
    
    # Return the configured OpenAI client
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        http_client=http_client,
        timeout=timeout,
        max_retries=3
    )

def create_openai_client(api_key: str, proxy: str = None, timeout: int = 60):
    """
    Create a sync client with robust error handling and connection resilience.
    
    Args:
        api_key: OpenAI API key
        proxy: Optional proxy URL
        timeout: Request timeout in seconds
        
    Returns:
        OpenAI client instance
    """
    # Set up httpx timeout settings
    timeout_settings = httpx.Timeout(
        timeout,  # Default timeout
        connect=10.0,  # Connection timeout
        read=timeout,  # Read timeout
        write=timeout,  # Write timeout
        pool=5.0  # Pool timeout
    )
    
    # Configure limits
    limits = httpx.Limits(
        max_keepalive_connections=5,
        max_connections=10,
        keepalive_expiry=30.0  # 30 seconds
    )
    
    # Create common client settings
    client_settings = {
        "timeout": timeout_settings,
        "limits": limits,
        "follow_redirects": True,
        "http2": True
    }
    
    # Add proxy if provided
    if proxy:
        client_settings["proxies"] = {
            "http://": proxy,
            "https://": proxy
        }
    
    # Create the httpx client
    http_client = httpx.Client(**client_settings)
    
    # Return the configured OpenAI client
    return OpenAI(
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        http_client=http_client,
        timeout=timeout,
        max_retries=3
    )

# Decorator for retrying OpenAI API calls with exponential backoff
def retry_openai_api(max_retries=3, initial_delay=1, backoff_factor=2):
    """
    Decorator for retrying OpenAI API calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplicative factor for backoff
        
    Returns:
        Decorated function
    """
    retry_strategy = tenacity.retry(
        stop=tenacity.stop_after_attempt(max_retries),
        wait=tenacity.wait_exponential(multiplier=initial_delay, factor=backoff_factor),
        retry=tenacity.retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    
    return retry_strategy