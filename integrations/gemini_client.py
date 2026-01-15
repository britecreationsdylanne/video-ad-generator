"""
Google Gemini API integration for image generation (Nano Banana)
Using the official google-genai Python SDK
"""

import os
import time
import base64
from typing import Dict, Optional
from google import genai
from google.genai import types


class GeminiClient:
    """Wrapper for Google Gemini API (Nano Banana image generation)"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_AI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_AI_API_KEY environment variable not set")

        # Initialize the client with API key
        self.client = genai.Client(api_key=self.api_key)
        # Use Gemini 2.5 Flash Image for image generation (Nano Banana)
        self.default_model = os.getenv("DEFAULT_IMAGE_MODEL", "gemini-2.5-flash-image")

    def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        aspect_ratio: str = "16:9",
        image_size: str = "1K",
        number_of_images: int = 1,
    ) -> Dict:
        """
        Generate an image using Nano Banana (Gemini 2.5 Flash)

        Args:
            prompt: Image description prompt
            model: Model to use (default: gemini-2.5-flash-preview-04-17)
            aspect_ratio: Not used for Nano Banana (kept for compatibility)
            image_size: Not used for Nano Banana (kept for compatibility)
            number_of_images: Not used for Nano Banana (kept for compatibility)

        Returns:
            {
                "image_data": "base64_encoded_image",
                "prompt": "original prompt",
                "model": "model-used",
                "cost_estimate": "$0.039",
                "generation_time_ms": 1234
            }
        """
        if not self.api_key:
            raise ValueError("Google AI API key not configured")

        model_name = model or self.default_model
        start_time = time.time()

        try:
            # Use Nano Banana for image generation
            print(f"[NANO BANANA] Using model: {model_name}")
            print(f"[NANO BANANA] Prompt: {prompt[:100]}...")

            # Generate image using generate_content
            response = self.client.models.generate_content(
                model=model_name,
                contents=[prompt]  # MUST be a list!
            )

            generation_time_ms = int((time.time() - start_time) * 1000)

            # Debug: Print response structure
            print(f"[NANO BANANA DEBUG] Response received")
            print(f"[NANO BANANA DEBUG] Response type: {type(response)}")

            # Handle different response formats based on google-genai version
            parts = []
            if hasattr(response, 'parts'):
                parts = response.parts
            elif hasattr(response, 'candidates') and response.candidates:
                # Newer API format
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts

            print(f"[NANO BANANA DEBUG] Number of parts: {len(parts)}")

            # Extract image data from response parts using part.as_image()
            image_data = None

            for i, part in enumerate(parts):
                print(f"[NANO BANANA DEBUG] Part {i}: has inline_data = {hasattr(part, 'inline_data')}, has text = {hasattr(part, 'text')}")

                # Use the as_image() method to get Image object (per documentation)
                if hasattr(part, 'inline_data') and part.inline_data:
                    try:
                        # as_image() returns a google.genai.types.Image object
                        image_obj = part.as_image()
                        print(f"[NANO BANANA DEBUG] Got Image object: {type(image_obj)}")

                        # The Image object has a _pil_image attribute for the actual PIL Image
                        if hasattr(image_obj, '_pil_image'):
                            pil_image = image_obj._pil_image
                            print(f"[NANO BANANA DEBUG] Got PIL Image from _pil_image: {type(pil_image)}, size: {pil_image.size}")

                            # Convert PIL Image to base64
                            from io import BytesIO
                            buffer = BytesIO()
                            pil_image.save(buffer, format='PNG')
                            image_bytes = buffer.getvalue()
                            image_data = base64.b64encode(image_bytes).decode('utf-8')

                            print(f"[NANO BANANA DEBUG] Image converted successfully, base64 size: {len(image_data)} bytes")
                            break
                        else:
                            print(f"[NANO BANANA ERROR] Image object has no _pil_image attribute")
                            print(f"[NANO BANANA ERROR] Available attributes: {[a for a in dir(image_obj) if not a.startswith('__')]}")
                    except Exception as img_error:
                        print(f"[NANO BANANA ERROR] Failed to convert image: {img_error}")
                        import traceback
                        traceback.print_exc()

            if not image_data:
                print(f"[NANO BANANA ERROR] No image data found in response")
                print(f"[NANO BANANA ERROR] Response parts count: {len(parts)}")
                if len(parts) > 0:
                    for i, part in enumerate(parts):
                        print(f"[NANO BANANA ERROR] Part {i} has text: {part.text[:200] if hasattr(part, 'text') and part.text else 'None'}")
                raise ValueError("No image data in response")

            # Cost estimate for Nano Banana ($30 per 1M tokens, 1290 tokens per image = ~$0.039)
            cost_per_image = 0.039
            cost_estimate = cost_per_image * number_of_images

            return {
                "image_data": image_data,  # Base64 encoded PNG
                "prompt": prompt,
                "model": model_name,
                "cost_estimate": f"${cost_estimate:.2f}",
                "generation_time_ms": generation_time_ms
            }

        except Exception as e:
            print(f"[NANO BANANA ERROR] Image generation failed: {str(e)}")
            print(f"[NANO BANANA ERROR] Model: {model_name}, Prompt: {prompt[:100]}...")
            import traceback
            traceback.print_exc()
            raise


# Singleton instance
_gemini_client = None


def get_gemini_client() -> GeminiClient:
    """Get or create Gemini client singleton"""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client
