"""
BriteCo Ad Generator API Server
Provides AI-powered endpoints for ad generation
Uses same proven architecture as venue-newsletter-tool
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Runway API configuration
RUNWAY_API_KEY = os.getenv('RUNWAY_API_KEY')
RUNWAY_API_BASE = 'https://api.dev.runwayml.com/v1'

# Google Veo API configuration
GOOGLE_VEO_API_KEY = os.getenv('GOOGLE_VEO_API_KEY')
VEO_API_BASE = 'https://generativelanguage.googleapis.com/v1beta'

# Import from local integrations folder
from integrations.openai_client import OpenAIClient
from integrations.gemini_client import GeminiClient
from integrations.claude_client import ClaudeClient

app = Flask(__name__, static_folder='.')
CORS(app)

# Initialize AI clients
openai_client = OpenAIClient()
gemini_client = GeminiClient()
claude_client = ClaudeClient()

print("[OK] OpenAI initialized")
print("[OK] Gemini initialized")
print("[OK] Claude initialized")

# BriteCo brand guidelines
BRAND_GUIDELINES = """
BriteCo Brand Guidelines:
- Colors: Turquoise (#31D7CA), Navy (#272D3F), Orange (#FC883A)
- Style: Modern, clean, optimistic, trustworthy
- Target: Millennials and Gen Z engaged couples
- Photography: Warm lighting, diverse couples, genuine moments
- No gradients - solid colors only
- Gilroy font family

Requirements for ads:
- Show happy couple with engagement ring or jewelry
- Warm, natural lighting
- Modern, clean aesthetic
- Include turquoise color accent somewhere
- Professional photography quality
- Authentic, candid moment (not too posed)
- Diverse representation
"""

# Platform specifications
PLATFORM_SIZES = {
    'meta': [
        {'name': 'Square Feed', 'width': 1080, 'height': 1080},
        {'name': 'Portrait Story', 'width': 1080, 'height': 1920},
        {'name': 'Landscape Link', 'width': 1200, 'height': 628}
    ],
    'reddit': [
        {'name': 'Feed', 'width': 1200, 'height': 628},
        {'name': 'Square', 'width': 960, 'height': 960}
    ],
    'pinterest': [
        {'name': 'Standard Pin', 'width': 1000, 'height': 1500},
        {'name': 'Square', 'width': 1000, 'height': 1000}
    ],
    'demandgen': [
        {'name': 'Landscape', 'width': 1200, 'height': 628},
        {'name': 'Square', 'width': 1200, 'height': 1200},
        {'name': 'Portrait', 'width': 960, 'height': 1200},
        {'name': 'Vertical', 'width': 1080, 'height': 1920}
    ],
    'pmax': [
        {'name': 'Landscape', 'width': 1200, 'height': 628},
        {'name': 'Square', 'width': 1200, 'height': 1200},
        {'name': 'Portrait', 'width': 960, 'height': 1200}
    ],
    'general': [
        {'name': 'Portrait', 'width': 1080, 'height': 1920}
    ]
}

# Google Ads Creative Best Practices (for Demand Gen and Performance Max)
GOOGLE_ADS_BEST_PRACTICES = """
GOOGLE ADS CREATIVE BEST PRACTICES:
- Use high-quality, clear images with the subject centered in 80% of the frame
- Minimal text overlay (less than 20% of image area)
- Effective lighting and professional composition
- Show the product/service clearly and prominently
- Use brand colors consistently but don't overwhelm
- Avoid cluttered backgrounds - clean, simple compositions work best
- Create visually compelling imagery that inspires action
- Feature aspirational lifestyle imagery showing jewelry in context
- Ensure images work well at smaller sizes (mobile-first design)
- Use contrasting colors to make key elements stand out
- Include clear visual hierarchy with one focal point
"""

@app.route('/')
def serve_index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/video_generator.html')
def serve_video_generator():
    """Serve the video generator page"""
    return send_from_directory('.', 'video_generator.html')

@app.route('/logo file for claude.jpg')
def serve_logo():
    """Serve the logo file"""
    return send_from_directory('.', 'logo file for claude.jpg')

@app.route('/logos/<filename>')
def serve_logo_file(filename):
    """Serve logo files from logos subdirectory"""
    return send_from_directory('logos', filename)

@app.route('/<filename>')
def serve_root_file(filename):
    """Serve logo files from root directory (PNG files)"""
    if filename.endswith('.png') or filename.endswith('.jpg'):
        return send_from_directory('.', filename)
    return "Not found", 404

@app.route('/api/generate-prompt', methods=['POST'])
def generate_prompt():
    """Generate image prompt using Claude or OpenAI"""
    try:
        data = request.json
        campaign_text = data.get('campaignText', '')
        platforms = data.get('platforms', [])
        provider = data.get('provider', 'claude')

        print(f"\n[API] Generate Prompt Request")
        print(f"  Provider: {provider}")
        print(f"  Platforms: {platforms}")

        # Check if Google Ads platforms are selected
        is_google_ads = any(p.lower() in ['demandgen', 'pmax'] for p in platforms)

        # Build platform-specific context
        if is_google_ads:
            platform_context = f"""You are an expert at creating image generation prompts for Google Ads campaigns.

Create an image generation prompt for BriteCo jewelry insurance ads for {', '.join(platforms)}.

Campaign context: {campaign_text}

{BRAND_GUIDELINES}

{GOOGLE_ADS_BEST_PRACTICES}

IMPORTANT FOR GOOGLE ADS:
- Images must be high-quality and work across YouTube, Discover, Gmail, and Display Network
- Subject should be centered in 80% of frame space
- Keep text overlay minimal (under 20% of image)
- Create aspirational lifestyle imagery that inspires action
- Ensure the imagery works well at smaller mobile sizes
- Use clear visual hierarchy with one strong focal point

Generate ONE detailed, creative prompt (200 words max) for Nano Banana (Google Gemini) image generator. Make it specific, visual, and actionable."""
        else:
            platform_context = f"""You are an expert at creating image generation prompts for AI models.

Create an image generation prompt for BriteCo jewelry insurance ads for {', '.join(platforms)}.

Campaign context: {campaign_text}

{BRAND_GUIDELINES}

Generate ONE detailed, creative prompt (200 words max) for Nano Banana (Google Gemini) image generator. Make it specific, visual, and actionable."""

        prompt_context = platform_context

        # Use selected provider
        if provider == 'claude':
            print("[API] Using Claude...")
            result = claude_client.generate_content(
                prompt=prompt_context,
                max_tokens=500,
                temperature=0.7
            )
            prompt = result.get('content', '')
        else:
            print("[API] Using OpenAI...")
            result = openai_client.generate_content(
                prompt=prompt_context,
                max_tokens=500,
                temperature=0.7
            )
            prompt = result.get('content', '')

        print(f"[API] Prompt generated successfully ({len(prompt)} chars)")

        return jsonify({
            'success': True,
            'prompt': prompt,
            'provider': provider
        })

    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate-images', methods=['POST'])
def generate_images():
    """Generate images using Gemini (Nano Banana) - 2 variations per size"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        platforms = data.get('platforms', [])

        print(f"\n[API] Generate Images Request")
        print(f"  Platforms: {platforms}")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Generating 2 variations per size")

        images = []

        for platform in platforms:
            platform_lower = platform.lower()
            sizes = PLATFORM_SIZES.get(platform_lower, PLATFORM_SIZES['meta'])

            for size in sizes:
                # Generate 2 variations for each size
                for variation_num in range(1, 3):  # 1, 2
                    try:
                        print(f"[API] Generating {platform} - {size['name']} - Variation {variation_num}...")

                        # Calculate aspect ratio for this size
                        width = size['width']
                        height = size['height']
                        aspect_ratio = width / height

                        # Determine aspect ratio string for Gemini prompt enhancement
                        if aspect_ratio > 1.5:
                            aspect_hint = "wide landscape format (16:9 or wider)"
                            composition_hint = "horizontal composition with subjects positioned to fill the wide frame"
                        elif aspect_ratio > 1.2:
                            aspect_hint = "landscape format"
                            composition_hint = "horizontal composition"
                        elif aspect_ratio > 0.85:
                            aspect_hint = "square format (1:1)"
                            composition_hint = "centered composition with subjects filling the square frame"
                        elif aspect_ratio > 0.6:
                            aspect_hint = "portrait format (4:5)"
                            composition_hint = "vertical composition with more headroom"
                        else:
                            aspect_hint = "tall portrait format (9:16 story)"
                            composition_hint = "full vertical composition from head to below waist, story-style framing"

                        # Enhance prompt with aspect ratio guidance
                        enhanced_prompt = f"{prompt}\n\nIMPORTANT: Compose this image specifically for {aspect_hint}. Use {composition_hint}. Frame: {width}x{height}px.\n\nDo NOT include any company logos, brand marks, watermarks, or text overlays in the image. Generate photography only without any branding elements."

                        # Generate with Gemini (Nano Banana) with aspect-specific prompt
                        result = gemini_client.generate_image(
                            prompt=enhanced_prompt,
                            model="gemini-2.5-flash-image"
                        )

                        image_data = result.get('image_data', '')

                        if image_data:
                            # Resize and compress image to reduce size
                            try:
                                import base64
                                from PIL import Image, ImageOps
                                from io import BytesIO

                                # Decode base64 to PIL Image
                                image_bytes = base64.b64decode(image_data)
                                pil_image = Image.open(BytesIO(image_bytes))

                                print(f"[API] Original image: {pil_image.size}")

                                # Use ImageOps.fit to crop/resize maintaining aspect ratio with centering
                                target_width = size['width']
                                target_height = size['height']
                                pil_image = ImageOps.fit(pil_image, (target_width, target_height), Image.Resampling.LANCZOS, centering=(0.5, 0.5))

                                # Convert to JPEG with compression to reduce size
                                buffer = BytesIO()
                                pil_image.convert('RGB').save(buffer, format='JPEG', quality=85, optimize=True)
                                compressed_bytes = buffer.getvalue()
                                image_data = base64.b64encode(compressed_bytes).decode('utf-8')

                                print(f"[API] Final: {pil_image.size}, compressed from {len(image_bytes)} to {len(compressed_bytes)} bytes")
                            except Exception as resize_error:
                                print(f"[API] WARNING - Resize failed, using original: {resize_error}")

                            images.append({
                                'platform': platform,
                                'size': f"{size['name']} - Variation {variation_num}",
                                'width': size['width'],
                                'height': size['height'],
                                'url': f"data:image/jpeg;base64,{image_data}"
                            })
                            print(f"[API] SUCCESS - {platform} {size['name']} - Variation {variation_num}")
                        else:
                            print(f"[API] WARNING - No image data for {platform} {size['name']} - Variation {variation_num}")

                    except Exception as img_error:
                        print(f"[API ERROR] Failed to generate {platform} {size['name']} - Variation {variation_num}: {img_error}")

        print(f"[API] Generated {len(images)} images total")

        return jsonify({
            'success': True,
            'images': images,
            'generated_at': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate-ad-copy', methods=['POST'])
def generate_ad_copy():
    """Generate platform-specific ad copy using Claude or OpenAI"""
    try:
        data = request.json
        platform = data.get('platform', '').lower()
        size_name = data.get('sizeName', '')
        campaign_text = data.get('campaignText', '')
        text_overlay = data.get('textOverlay', '')
        provider = data.get('provider', 'claude')

        print(f"\n[API] Generate Ad Copy Request")
        print(f"  Platform: {platform}, Size: {size_name}")
        print(f"  Provider: {provider}")

        # Platform-specific specs
        platform_specs = {
            'meta': {
                'headline_limit': 27,
                'primary_text_visible': 125,
                'description_limit': 27,
                'best_practices': 'Front-load value proposition in first 30 characters. Use emojis sparingly.'
            },
            'reddit': {
                'headline_limit': 300,
                'body_limit': 500,
                'best_practices': 'Be authentic and conversational. Redditors value transparency and community.'
            },
            'pinterest': {
                'title_limit': 100,
                'title_visible': 40,
                'description_limit': 500,
                'best_practices': 'Focus on aspirational, visual language. Pinterest is about inspiration and discovery.'
            }
        }

        specs = platform_specs.get(platform, platform_specs['meta'])

        # Build prompt based on platform
        if platform == 'meta':
            prompt_context = f"""You are an expert social media copywriter for BriteCo jewelry insurance.

Generate ad copy for a {platform.upper()} ad campaign.

Campaign context: {campaign_text}
Text overlay on image: "{text_overlay}"
Ad size: {size_name}

Platform specifications for {platform.upper()}:
{', '.join([f'{k}: {v}' for k, v in specs.items()])}

BriteCo Brand Voice:
- Modern, trustworthy, optimistic
- Target: Millennials and Gen Z engaged couples
- Focus on peace of mind and protecting what matters
- Turquoise (#31D7CA), Navy (#272D3F), Orange (#FC883A) brand colors

Generate:
1. Headline (stay within 27 characters)
2. Primary text (engaging, benefit-focused, first 125 chars are most visible)
3. Description (stay within 27 characters, appears below primary text)
4. Call-to-action suggestion

Return as JSON:
{{
  "headline": "...",
  "body": "...",
  "description": "...",
  "cta": "..."
}}

Return ONLY the JSON, no other text."""
        else:
            prompt_context = f"""You are an expert social media copywriter for BriteCo jewelry insurance.

Generate ad copy for a {platform.upper()} ad campaign.

Campaign context: {campaign_text}
Text overlay on image: "{text_overlay}"
Ad size: {size_name}

Platform specifications for {platform.upper()}:
{', '.join([f'{k}: {v}' for k, v in specs.items()])}

BriteCo Brand Voice:
- Modern, trustworthy, optimistic
- Target: Millennials and Gen Z engaged couples
- Focus on peace of mind and protecting what matters
- Turquoise (#31D7CA), Navy (#272D3F), Orange (#FC883A) brand colors

Generate:
1. Headline (stay within character limits)
2. Primary text/body copy (engaging, benefit-focused)
3. Call-to-action suggestion

Return as JSON:
{{
  "headline": "...",
  "body": "...",
  "cta": "..."
}}

Return ONLY the JSON, no other text."""

        # Use selected provider
        if provider == 'claude':
            print("[API] Using Claude...")
            result = claude_client.generate_content(
                prompt=prompt_context,
                max_tokens=500,
                temperature=0.7
            )
            copy_text = result.get('content', '')
        else:
            print("[API] Using OpenAI...")
            result = openai_client.generate_content(
                prompt=prompt_context,
                max_tokens=500,
                temperature=0.7
            )
            copy_text = result.get('content', '')

        # Parse JSON from response
        import json
        import re

        # Remove markdown code blocks if present
        copy_text = re.sub(r'```json\s*', '', copy_text)
        copy_text = re.sub(r'```\s*', '', copy_text)
        copy_text = copy_text.strip()

        try:
            ad_copy = json.loads(copy_text)
        except:
            # If JSON parsing fails, create default structure
            ad_copy = {
                'headline': copy_text[:100],
                'body': 'Protect what matters most with BriteCo jewelry insurance.',
                'cta': 'Get Protected'
            }

        print(f"[API] Ad copy generated successfully")

        return jsonify({
            'success': True,
            'adCopy': ad_copy,
            'provider': provider
        })

    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate-animation', methods=['POST'])
def generate_animation():
    """Generate animated GIF from multiple image variations"""
    try:
        data = request.json
        base_prompt = data.get('prompt', '')
        width = data.get('width', 1080)
        height = data.get('height', 1080)
        frame_count = data.get('frame_count', 5)
        fps = data.get('fps', 2)  # Frames per second (default: 2)
        platform = data.get('platform', 'Meta')
        size_name = data.get('sizeName', 'Square')

        # Ensure frame_count is within limits
        frame_count = min(max(frame_count, 3), 7)  # Between 3 and 7 frames

        print(f"\n[API] Generate Animation Request")
        print(f"  Platform: {platform}, Size: {size_name}")
        print(f"  Frames: {frame_count}, FPS: {fps}")
        print(f"  Dimensions: {width}x{height}")
        frames = []

        # Define prompt variations for animation effect
        variations = [
            "",  # Original
            "Zoom in slightly to show more detail on the jewelry",
            "Pan slightly to the right to show the couple's connection",
            "Zoom out slightly for a wider view",
            "Different angle showing the couple's happy expressions",
            "Close-up on the ring with couple softly blurred in background",
            "Return to original composition with warm lighting"
        ]

        for i in range(frame_count):
            try:
                variation_prompt = variations[i] if i < len(variations) else ""
                enhanced_prompt = f"{base_prompt}\n\n{variation_prompt}".strip()

                print(f"[API] Generating frame {i+1}/{frame_count}...")

                # Calculate aspect ratio hints
                aspect_ratio = width / height
                if aspect_ratio > 1.5:
                    aspect_hint = "wide landscape format (16:9 or wider)"
                    composition_hint = "horizontal composition with subjects positioned to fill the wide frame"
                elif aspect_ratio > 1.2:
                    aspect_hint = "landscape format"
                    composition_hint = "horizontal composition"
                elif aspect_ratio > 0.85:
                    aspect_hint = "square format (1:1)"
                    composition_hint = "centered composition with subjects filling the square frame"
                elif aspect_ratio > 0.6:
                    aspect_hint = "portrait format (4:5)"
                    composition_hint = "vertical composition with more headroom"
                else:
                    aspect_hint = "tall portrait format (9:16 story)"
                    composition_hint = "full vertical composition from head to below waist, story-style framing"

                enhanced_prompt += f"\n\nIMPORTANT: Compose this image specifically for {aspect_hint}. Use {composition_hint}. Frame: {width}x{height}px.\n\nDo NOT include any company logos, brand marks, watermarks, or text overlays in the image. Generate photography only without any branding elements."

                # Generate frame with Gemini
                result = gemini_client.generate_image(
                    prompt=enhanced_prompt,
                    model="gemini-2.5-flash-image"
                )

                image_data = result.get('image_data', '')

                if image_data:
                    import base64
                    from PIL import Image, ImageOps
                    from io import BytesIO

                    # Decode and resize frame
                    image_bytes = base64.b64decode(image_data)
                    pil_image = Image.open(BytesIO(image_bytes))

                    # Resize to exact dimensions
                    pil_image = ImageOps.fit(pil_image, (width, height), Image.Resampling.LANCZOS, centering=(0.5, 0.5))

                    frames.append(pil_image.convert('RGB'))
                    print(f"[API] Frame {i+1} generated successfully")

            except Exception as frame_error:
                print(f"[API ERROR] Failed to generate frame {i+1}: {frame_error}")

        if len(frames) == 0:
            return jsonify({'success': False, 'error': 'No frames generated'}), 500

        print(f"[API] Creating GIF from {len(frames)} frames...")

        # Create animated GIF
        from io import BytesIO
        import base64

        # Convert frames to base64 for individual editing
        frame_images = []
        for i, frame in enumerate(frames):
            frame_buffer = BytesIO()
            frame.save(frame_buffer, format='PNG')
            frame_base64 = base64.b64encode(frame_buffer.getvalue()).decode('utf-8')
            frame_images.append(f"data:image/png;base64,{frame_base64}")

        gif_buffer = BytesIO()

        # Calculate duration per frame in milliseconds based on FPS
        duration_ms = int(1000 / fps)

        print(f"[API] Creating GIF with {len(frames)} frames at {fps} FPS ({duration_ms}ms per frame)")

        # Save as animated GIF
        frames[0].save(
            gif_buffer,
            format='GIF',
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,  # Infinite loop
            optimize=True
        )

        gif_data = base64.b64encode(gif_buffer.getvalue()).decode('utf-8')

        total_duration = len(frames) / fps
        print(f"[API] Animation created successfully ({len(gif_buffer.getvalue())} bytes, {total_duration:.1f}s total)")

        return jsonify({
            'success': True,
            'animation': f"data:image/gif;base64,{gif_data}",
            'frames': frame_images,  # Individual frames for editing
            'frame_count': len(frames),
            'fps': fps
        })

    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# VIDEO PROMPT GENERATION (Claude/OpenAI)
# ============================================================================

@app.route('/api/generate-video-prompt', methods=['POST'])
def generate_video_prompt():
    """Generate video prompt using Claude or OpenAI"""
    try:
        data = request.json
        campaign_text = data.get('campaignText', '')
        platform = data.get('platform', 'tiktok')
        duration = data.get('duration', 6)
        motion_style = data.get('motionStyle', 'smooth')
        provider = data.get('provider', 'claude')

        print(f"\n[API] Generate Video Prompt Request")
        print(f"  Provider: {provider}")
        print(f"  Platform: {platform}")
        print(f"  Duration: {duration}s")
        print(f"  Motion: {motion_style}")

        # Camera motion descriptions
        motion_descriptions = {
            'smooth': 'smooth, fluid camera movement with gentle pans',
            'zoom': 'gradual zoom revealing details',
            'dynamic': 'dynamic camera angles with elegant movement'
        }
        motion_desc = motion_descriptions.get(motion_style, motion_descriptions['smooth'])

        prompt_context = f"""You are an expert at creating video generation prompts for AI video models like Google Veo.

Create a video generation prompt for a BriteCo jewelry insurance advertisement.

Campaign context: {campaign_text}

Platform: {platform}
Video Duration: {duration} seconds
Camera Style: {motion_desc}

{BRAND_GUIDELINES}

VEO VIDEO BEST PRACTICES:
- Structure: Subject + Action + Style + Lighting
- Use clear, action-focused descriptions
- Avoid negative words like "no" or "don't"
- Describe camera movement explicitly
- Include lighting and atmosphere details
- Keep it under 500 words for best results

Create ONE detailed, cinematic video prompt that:
1. Opens with an establishing shot of jewelry
2. Describes the camera movement and action
3. Captures the emotional, aspirational feeling of protecting precious jewelry
4. Ends with a memorable visual moment

The video should feel like a premium luxury brand commercial that makes viewers feel the importance of protecting their valuable jewelry."""

        # Use selected provider
        if provider == 'claude':
            print("[API] Using Claude for video prompt...")
            result = claude_client.generate_content(
                prompt=prompt_context,
                max_tokens=800,
                temperature=0.8
            )
            prompt = result.get('content', '')
        else:
            print("[API] Using OpenAI for video prompt...")
            result = openai_client.generate_content(
                prompt=prompt_context,
                max_tokens=800,
                temperature=0.8
            )
            prompt = result.get('content', '')

        print(f"[API] Video prompt generated successfully ({len(prompt)} chars)")

        return jsonify({
            'success': True,
            'prompt': prompt,
            'provider': provider
        })

    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# VIDEO GENERATION (Veo API - Default, Runway API - Fallback)
# ============================================================================

@app.route('/api/generate-video', methods=['POST'])
def generate_video():
    """Generate video using Google Veo API (default) or Runway API"""
    try:
        data = request.json
        prompt_text = data.get('prompt', '')
        prompt_image = data.get('image')  # Base64 data URI or URL
        duration = int(data.get('duration', 8))  # Must be 4, 6, or 8
        aspect_ratio = data.get('ratio', '720:1280')  # Default 9:16 portrait
        provider = data.get('provider', 'veo')  # 'veo' or 'runway'

        print(f"\n[API] Generate Video Request")
        print(f"  Provider: {provider}")
        print(f"  Duration: {duration}s")
        print(f"  Ratio: {aspect_ratio}")
        print(f"  Prompt: {prompt_text[:100]}...")
        print(f"  Has Image: {bool(prompt_image)}")

        # Route to appropriate provider
        if provider == 'veo':
            return generate_video_veo(prompt_text, prompt_image, duration, aspect_ratio)
        else:
            return generate_video_runway(prompt_text, prompt_image, duration, aspect_ratio)

    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def generate_video_veo(prompt_text, prompt_image, duration, aspect_ratio):
    """Generate video using Google Veo 3.1 API"""
    if not GOOGLE_VEO_API_KEY:
        return jsonify({'success': False, 'error': 'Google Veo API key not configured'}), 500

    # Convert aspect ratio format (720:1280 -> 9:16)
    ratio_map = {
        '720:1280': '9:16',
        '1280:720': '16:9',
        '1080:1920': '9:16',
        '1920:1080': '16:9'
    }
    veo_aspect_ratio = ratio_map.get(aspect_ratio, '9:16')

    # Ensure duration is valid integer (4, 6, or 8)
    valid_durations = [4, 6, 8]
    if duration not in valid_durations:
        duration = 8  # Default to 8 if invalid

    # Veo API endpoint
    model = 'veo-3.1-generate-preview'
    endpoint = f'{VEO_API_BASE}/models/{model}:predictLongRunning'

    headers = {
        'Content-Type': 'application/json',
        'x-goog-api-key': GOOGLE_VEO_API_KEY
    }

    # Build payload for Veo API
    # IMPORTANT: durationSeconds must be an integer, not a string
    payload = {
        'instances': [{
            'prompt': prompt_text
        }],
        'parameters': {
            'aspectRatio': veo_aspect_ratio,
            'durationSeconds': duration,  # Integer, not string!
            'resolution': '720p',
            'sampleCount': 1
        }
    }

    # Add image for image-to-video
    if prompt_image:
        # Extract base64 data and determine mime type
        if ',' in prompt_image:
            # Data URI format: data:image/jpeg;base64,xxxxx
            header, base64_data = prompt_image.split(',', 1)
            # Extract mime type from header
            if 'png' in header.lower():
                mime_type = 'image/png'
            elif 'gif' in header.lower():
                mime_type = 'image/gif'
            else:
                mime_type = 'image/jpeg'
        else:
            base64_data = prompt_image
            mime_type = 'image/jpeg'

        # Veo 3.1 image-to-video format
        payload['instances'][0]['image'] = {
            'bytesBase64Encoded': base64_data,
            'mimeType': mime_type
        }

    print(f"[API] Calling Google Veo API: {endpoint}")
    print(f"[API] Payload: {json.dumps({**payload, 'instances': [{'prompt': prompt_text[:50] + '...'}]}, indent=2)}")

    # Create the task
    response = requests.post(endpoint, headers=headers, json=payload)

    print(f"[API] Veo Response Status: {response.status_code}")

    if response.status_code != 200:
        error_msg = response.text
        print(f"[API ERROR] Veo API error: {error_msg}")
        return jsonify({'success': False, 'error': f'Veo API error: {error_msg}'}), 500

    task_data = response.json()
    # Veo returns operation name like "operations/xxx"
    operation_name = task_data.get('name', '')

    print(f"[API] Veo Task created: {operation_name}")

    return jsonify({
        'success': True,
        'task_id': operation_name,
        'provider': 'veo',
        'status': 'processing',
        'message': 'Video generation started with Google Veo 3.1'
    })


def generate_video_runway(prompt_text, prompt_image, duration, aspect_ratio):
    """Generate video using Runway API (fallback)"""
    if not RUNWAY_API_KEY:
        return jsonify({'success': False, 'error': 'Runway API key not configured'}), 500

    headers = {
        'Authorization': f'Bearer {RUNWAY_API_KEY}',
        'Content-Type': 'application/json',
        'X-Runway-Version': '2024-11-06'
    }

    # Add image if provided (for image-to-video)
    if prompt_image:
        payload = {
            'model': 'gen4_turbo',
            'promptText': prompt_text,
            'promptImage': prompt_image,
            'ratio': aspect_ratio,
            'duration': duration
        }
        endpoint = f'{RUNWAY_API_BASE}/image_to_video'
    else:
        payload = {
            'model': 'veo3.1',
            'promptText': prompt_text,
            'ratio': aspect_ratio,
            'duration': 8
        }
        endpoint = f'{RUNWAY_API_BASE}/text_to_video'

    print(f"[API] Calling Runway API: {endpoint}")
    print(f"[API] Payload: {json.dumps(payload, indent=2)}")

    response = requests.post(endpoint, headers=headers, json=payload)

    if response.status_code != 200 and response.status_code != 201:
        error_msg = response.text
        print(f"[API ERROR] Runway API error: {error_msg}")
        return jsonify({'success': False, 'error': f'Runway API error: {error_msg}'}), 500

    task_data = response.json()
    task_id = task_data.get('id')

    print(f"[API] Runway Task created: {task_id}")

    return jsonify({
        'success': True,
        'task_id': task_id,
        'provider': 'runway',
        'status': 'processing',
        'message': 'Video generation started with Runway'
    })


@app.route('/api/video-status/<path:task_id>', methods=['GET'])
def get_video_status(task_id):
    """Check status of a video generation task (Veo or Runway)"""
    try:
        provider = request.args.get('provider', 'veo')

        if provider == 'veo':
            return get_veo_status(task_id)
        else:
            return get_runway_status(task_id)

    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def get_veo_status(operation_name):
    """Check status of a Google Veo video generation task"""
    if not GOOGLE_VEO_API_KEY:
        return jsonify({'success': False, 'error': 'Google Veo API key not configured'}), 500

    # Poll the operation status
    endpoint = f'{VEO_API_BASE}/{operation_name}'

    headers = {
        'x-goog-api-key': GOOGLE_VEO_API_KEY
    }

    response = requests.get(endpoint, headers=headers)

    if response.status_code != 200:
        return jsonify({'success': False, 'error': f'Failed to get Veo task status: {response.text}'}), 500

    task_data = response.json()
    done = task_data.get('done', False)

    result = {
        'success': True,
        'task_id': operation_name,
        'provider': 'veo'
    }

    if done:
        # Check for error
        if 'error' in task_data:
            result['status'] = 'FAILED'
            result['error'] = task_data['error'].get('message', 'Unknown error')
            print(f"[API] Veo video generation failed: {result['error']}")
        else:
            result['status'] = 'SUCCEEDED'
            # Extract video URL from response
            response_data = task_data.get('response', {})
            generated_samples = response_data.get('generateVideoResponse', {}).get('generatedSamples', [])
            if generated_samples:
                video_uri = generated_samples[0].get('video', {}).get('uri', '')
                # The Veo URL requires auth, so we proxy it through our server
                # Extract the file ID from the URI and create a proxy URL
                if 'files/' in video_uri:
                    file_id = video_uri.split('files/')[1].split(':')[0]
                    result['video_url'] = f'/api/veo-video/{file_id}'
                    result['original_url'] = video_uri
                else:
                    result['video_url'] = video_uri
            print(f"[API] Veo video generation complete: {result.get('video_url', 'no URL')}")
    else:
        result['status'] = 'RUNNING'
        # Veo doesn't provide progress percentage, estimate based on time
        metadata = task_data.get('metadata', {})
        result['progress'] = 50  # Placeholder
        print(f"[API] Veo video generation in progress...")

    return jsonify(result)


@app.route('/api/veo-video/<file_id>')
def proxy_veo_video(file_id):
    """Proxy endpoint to serve Veo-generated videos (requires API key auth)"""
    try:
        if not GOOGLE_VEO_API_KEY:
            return jsonify({'success': False, 'error': 'Google Veo API key not configured'}), 500

        # Build the download URL
        download_url = f'{VEO_API_BASE}/files/{file_id}:download?alt=media'

        headers = {
            'x-goog-api-key': GOOGLE_VEO_API_KEY
        }

        print(f"[API] Proxying Veo video: {download_url}")

        # Stream the video from Google
        response = requests.get(download_url, headers=headers, stream=True)

        if response.status_code != 200:
            print(f"[API ERROR] Failed to download Veo video: {response.status_code} - {response.text[:200]}")
            return jsonify({'success': False, 'error': f'Failed to download video: {response.status_code}'}), 500

        # Get content type from response
        content_type = response.headers.get('Content-Type', 'video/mp4')

        print(f"[API] Streaming Veo video, Content-Type: {content_type}")

        # Stream the response to the client
        from flask import Response
        return Response(
            response.iter_content(chunk_size=8192),
            content_type=content_type,
            headers={
                'Content-Disposition': f'inline; filename="veo_video_{file_id}.mp4"',
                'Cache-Control': 'public, max-age=3600'
            }
        )

    except Exception as e:
        print(f"[API ERROR] Veo video proxy error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def get_runway_status(task_id):
    """Check status of a Runway video generation task"""
    if not RUNWAY_API_KEY:
        return jsonify({'success': False, 'error': 'Runway API key not configured'}), 500

    headers = {
        'Authorization': f'Bearer {RUNWAY_API_KEY}',
        'X-Runway-Version': '2024-11-06'
    }

    response = requests.get(f'{RUNWAY_API_BASE}/tasks/{task_id}', headers=headers)

    if response.status_code != 200:
        return jsonify({'success': False, 'error': f'Failed to get task status: {response.text}'}), 500

    task_data = response.json()
    status = task_data.get('status', 'unknown')

    result = {
        'success': True,
        'task_id': task_id,
        'status': status,
        'provider': 'runway'
    }

    # If complete, include the output URL
    if status == 'SUCCEEDED':
        output = task_data.get('output', [])
        if output and len(output) > 0:
            result['video_url'] = output[0]
        print(f"[API] Runway video generation complete: {result.get('video_url', 'no URL')}")
    elif status == 'FAILED':
        result['error'] = task_data.get('failure', 'Unknown error')
        print(f"[API] Runway video generation failed: {result['error']}")
    else:
        # Still processing
        progress = task_data.get('progress', 0)
        result['progress'] = progress
        print(f"[API] Runway video generation progress: {progress}%")

    return jsonify(result)


@app.route('/api/generate-video-sync', methods=['POST'])
def generate_video_sync():
    """Generate video and wait for completion (synchronous)"""
    try:
        data = request.json
        prompt_text = data.get('prompt', '')
        prompt_image = data.get('image')
        duration = int(data.get('duration', 6))  # Must be int: 4, 6, or 8 for veo3
        aspect_ratio = data.get('ratio', '720:1280')

        print(f"\n[API] Generate Video (Sync) Request")
        print(f"  Duration: {duration}s")
        print(f"  Ratio: {aspect_ratio}")

        if not RUNWAY_API_KEY:
            return jsonify({'success': False, 'error': 'Runway API key not configured'}), 500

        headers = {
            'Authorization': f'Bearer {RUNWAY_API_KEY}',
            'Content-Type': 'application/json',
            'X-Runway-Version': '2024-11-06'
        }

        if prompt_image:
            # Image-to-video: supports gen4_turbo, gen3a_turbo, veo3, veo3.1, veo3.1_fast
            payload = {
                'model': 'gen4_turbo',
                'promptText': prompt_text,
                'promptImage': prompt_image,
                'ratio': aspect_ratio,
                'duration': duration
            }
            endpoint = f'{RUNWAY_API_BASE}/image_to_video'
        else:
            # Text-to-video: only supports veo3, veo3.1, veo3.1_fast
            # Using veo3.1 with duration=8 (must be 4, 6, or 8)
            payload = {
                'model': 'veo3.1',
                'promptText': prompt_text,
                'ratio': aspect_ratio,
                'duration': 8
            }
            endpoint = f'{RUNWAY_API_BASE}/text_to_video'

        # Create the task
        response = requests.post(endpoint, headers=headers, json=payload)

        if response.status_code not in [200, 201]:
            return jsonify({'success': False, 'error': f'Runway API error: {response.text}'}), 500

        task_data = response.json()
        task_id = task_data.get('id')
        print(f"[API] Task created: {task_id}, polling for completion...")

        # Poll for completion (max 5 minutes)
        max_attempts = 60
        for attempt in range(max_attempts):
            time.sleep(5)  # Wait 5 seconds between polls

            status_response = requests.get(f'{RUNWAY_API_BASE}/tasks/{task_id}', headers=headers)

            if status_response.status_code != 200:
                continue

            status_data = status_response.json()
            status = status_data.get('status', 'unknown')

            if status == 'SUCCEEDED':
                output = status_data.get('output', [])
                video_url = output[0] if output else None
                print(f"[API] Video complete: {video_url}")
                return jsonify({
                    'success': True,
                    'video_url': video_url,
                    'task_id': task_id
                })
            elif status == 'FAILED':
                error = status_data.get('failure', 'Unknown error')
                print(f"[API] Video failed: {error}")
                return jsonify({'success': False, 'error': error}), 500

            progress = status_data.get('progress', 0)
            print(f"[API] Progress: {progress}% (attempt {attempt + 1}/{max_attempts})")

        return jsonify({'success': False, 'error': 'Video generation timed out'}), 500

    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 3000))

    print("=" * 80)
    print("BriteCo Ad Generator - Python API Server")
    print("=" * 80)
    print(f"\nStarting server...")
    print(f"URL: http://localhost:{port}")
    print(f"\nAPIs:")
    print(f"  [OK] Claude (claude-sonnet-4-5-20250929)")
    print(f"  [OK] OpenAI (gpt-4o)")
    print(f"  [OK] Google Gemini (gemini-2.5-flash-image / Nano Banana)")
    print(f"  [OK] Google Veo 3.1 (video generation - DEFAULT)" if GOOGLE_VEO_API_KEY else "  [--] Google Veo (not configured)")
    print(f"  [OK] Runway (video generation - fallback)" if RUNWAY_API_KEY else "  [--] Runway (not configured)")
    print(f"\nPress Ctrl+C to stop")
    print("=" * 80)
    print()

    # Use PORT from environment (for Cloud Run) or default to 3000 for local dev
    app.run(debug=os.environ.get('FLASK_DEBUG', 'true').lower() == 'true', port=port, host='0.0.0.0')
