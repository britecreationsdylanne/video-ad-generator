"""
Claude (Anthropic) API Client
For content generation using Claude models
"""

import os
import time
from anthropic import Anthropic


class ClaudeClient:
    """Client for Claude API"""

    def __init__(self, api_key=None):
        """Initialize Claude client"""
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.client = Anthropic(api_key=self.api_key)
        self.default_model = "claude-sonnet-4-5-20250929"  # Claude Sonnet 4.5

    def generate_content(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        model: str = None
    ) -> dict:
        """
        Generate content using Claude

        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Creativity (0-1)
            max_tokens: Max response length
            model: Model to use (defaults to claude-3-5-sonnet)

        Returns:
            dict with content, model, tokens, cost_estimate, latency_ms
        """
        start_time = time.time()

        model_name = model or self.default_model

        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Call Claude API
        response = self.client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt if system_prompt else "",
            messages=messages
        )

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        # Extract content
        content = response.content[0].text

        # Calculate tokens
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens

        # Estimate cost
        cost_estimate = self._estimate_cost(model_name, input_tokens, output_tokens)

        return {
            "content": content,
            "model": model_name,
            "tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_estimate": f"${cost_estimate:.4f}",
            "latency_ms": latency_ms
        }

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on model pricing"""

        # Claude 3.5 Sonnet pricing (as of 2025)
        if "sonnet" in model.lower():
            input_cost = (input_tokens / 1_000_000) * 3.00  # $3 per 1M input tokens
            output_cost = (output_tokens / 1_000_000) * 15.00  # $15 per 1M output tokens
        # Claude 3 Haiku (cheaper option)
        elif "haiku" in model.lower():
            input_cost = (input_tokens / 1_000_000) * 0.25
            output_cost = (output_tokens / 1_000_000) * 1.25
        # Claude 3 Opus (premium)
        elif "opus" in model.lower():
            input_cost = (input_tokens / 1_000_000) * 15.00
            output_cost = (output_tokens / 1_000_000) * 75.00
        else:
            input_cost = (input_tokens / 1_000_000) * 3.00
            output_cost = (output_tokens / 1_000_000) * 15.00

        return input_cost + output_cost


# Singleton instance
_claude_client = None


def get_claude_client() -> ClaudeClient:
    """Get or create Claude client singleton"""
    global _claude_client
    if _claude_client is None:
        _claude_client = ClaudeClient()
    return _claude_client
