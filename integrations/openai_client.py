"""
OpenAI API integration for content generation
"""

import os
import time
from typing import Dict, List, Optional
from openai import OpenAI
import json


class OpenAIClient:
    """Wrapper for OpenAI API calls"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.default_model = os.getenv("DEFAULT_CONTENT_MODEL", "gpt-4o")

    def generate_content(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[List] = None,
    ) -> Dict:
        """
        Generate content using OpenAI

        Args:
            prompt: User prompt
            system_prompt: System instructions
            model: Model to use (default: gpt-4o)
            temperature: Creativity (0-2)
            max_tokens: Maximum response length
            tools: Optional function calling tools

        Returns:
            {
                "content": "generated text",
                "model": "model-used",
                "tokens": 123,
                "cost_estimate": "$0.05"
            }
        """
        if not self.client:
            raise ValueError("OpenAI API key not configured")

        model = model or self.default_model
        start_time = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build kwargs
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add tools if provided (for web search)
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)

        latency_ms = int((time.time() - start_time) * 1000)
        tokens_used = response.usage.total_tokens

        # Rough cost estimation (update with actual pricing)
        cost_per_1k_tokens = 0.03 if "gpt-4" in model else 0.002
        cost_estimate = (tokens_used / 1000) * cost_per_1k_tokens

        return {
            "content": response.choices[0].message.content,
            "model": model,
            "tokens": tokens_used,
            "cost_estimate": f"${cost_estimate:.4f}",
            "latency_ms": latency_ms,
            "finish_reason": response.choices[0].finish_reason,
            "raw_response": response,  # Include full response for tool calls
        }


# Singleton instance
_openai_client = None


def get_openai_client() -> OpenAIClient:
    """Get or create OpenAI client singleton"""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient()
    return _openai_client
