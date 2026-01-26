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
import subprocess
import tempfile
import base64
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, Response, send_file
from flask_cors import CORS
from dotenv import load_dotenv

# Google Auth for Vertex AI
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest

# Load environment variables
load_dotenv()

# Runway API configuration
RUNWAY_API_KEY = os.getenv('RUNWAY_API_KEY')
RUNWAY_API_BASE = 'https://api.dev.runwayml.com/v1'

# Google Veo API configuration (Vertex AI)
VEO_LOCATION = os.getenv('GOOGLE_CLOUD_REGION', 'us-central1')
VEO_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT', 'ai-tools-484419')
VEO_MODEL_ID = 'veo-3.1-generate-001'


def get_veo_auth_token():
    """Get OAuth2 access token for Vertex AI using Application Default Credentials"""
    try:
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        creds, project = google.auth.default(scopes=scopes)
        if not creds.valid:
            creds.refresh(GoogleAuthRequest())
        return creds.token
    except Exception as e:
        print(f"[VEO AUTH ERROR] Failed to get ADC token: {e}")
        return None

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
- Colors: Turquoise, Navy, Orange (use these colors visually, NOT as text)
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

CRITICAL: NEVER write, render, display, or include ANY text on the image including:
- Color codes (like #31D7CA, #272D3F, #FC883A)
- Hex values or RGB values
- Color names as visible text
- Any watermarks, labels, or text overlays
- Brand names or logos
Colors should ONLY be used visually in the image composition, never as readable text.
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

@app.route('/api/analyze-inspiration-images', methods=['POST'])
def analyze_inspiration_images():
    """Analyze inspiration images using Gemini Vision and return style descriptions"""
    try:
        data = request.json
        images = data.get('images', [])  # Array of base64 image data URIs

        print(f"\n[API] ========== INSPIRATION IMAGE ANALYSIS ==========")
        print(f"[API] Number of images received: {len(images)}")

        if not images:
            print(f"[API] WARNING: No images in request!")
            return jsonify({'success': True, 'analysis': ''})

        # Log info about each image
        for i, img in enumerate(images):
            if img:
                print(f"[API] Image {i+1}: data length = {len(img)}, starts with = {img[:50]}...")
            else:
                print(f"[API] Image {i+1}: EMPTY/NULL")

        # Use Gemini to analyze the images with structured creative brief format
        analysis_prompt = """Analyze this inspiration image and create a detailed creative brief for AI image generation.

Describe the following elements in detail:

**SUBJECT MATTER**: What is depicted in this image? Describe the main subjects, objects, characters, animals, or scenes shown. Be specific about what you actually see.

**COMPOSITION**: How is the image framed? Describe the layout, perspective, use of negative space, focal points, and visual hierarchy.

**STYLE CUES**: Is this photographic or illustrative? What era or aesthetic does it reference? Describe textures, rendering style, and artistic approach.

**COLOR PALETTE**: List the dominant colors (3-7 key colors). Describe the overall color mood (warm, cool, muted, vibrant, etc.).

**LIGHTING**: Describe the lighting quality - soft or hard, direction, mood it creates, shadows and highlights.

**MOOD/ATMOSPHERE**: What emotional tone does this image convey? What feelings does it evoke?

**TRANSFERABLE ATTRIBUTES**: List specific visual elements that can be safely incorporated into new images (poses, compositions, color schemes, lighting setups, etc.).

**PROMPT SNIPPETS**: Provide 3-5 short descriptive phrases that capture the essence of this image and could be directly used in an image generation prompt.

Be specific and detailed. Focus on what you actually see in the image - if it's a bear, describe the bear. If it's a landscape, describe the landscape. Your description will be used to guide AI image generation."""

        # Call Gemini with vision capability
        result = gemini_client.analyze_images(
            images=images,
            prompt=analysis_prompt,
            max_tokens=2000,
            temperature=0.3
        )

        analysis = result.get('content', '')
        print(f"[API] Image analysis complete ({len(analysis)} chars)")

        return jsonify({
            'success': True,
            'analysis': analysis
        })

    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/generate-prompt', methods=['POST'])
def generate_prompt():
    """Generate image prompt using Claude, OpenAI, or Gemini"""
    try:
        data = request.json
        campaign_text = data.get('campaignText', '')
        platforms = data.get('platforms', [])
        provider = data.get('provider', 'claude') or 'claude'  # Fallback if empty string
        inspiration_analysis = data.get('inspirationAnalysis', '')  # Style analysis from inspiration images

        print(f"\n[API] ========== GENERATE PROMPT REQUEST ==========")
        print(f"[API] Provider: {provider}")
        print(f"[API] Platforms: {platforms}")
        print(f"[API] Campaign text length: {len(campaign_text)}")
        print(f"[API] Has inspiration analysis: {bool(inspiration_analysis)}")
        if inspiration_analysis:
            print(f"[API] Inspiration analysis length: {len(inspiration_analysis)}")
            print(f"[API] Inspiration analysis preview: {inspiration_analysis[:200]}...")
        else:
            print(f"[API] WARNING: No inspiration analysis received!")

        # Check if Google Ads platforms are selected
        is_google_ads = any(p.lower() in ['demandgen', 'pmax'] for p in platforms)

        # Build inspiration context if available
        inspiration_context = ""
        if inspiration_analysis:
            inspiration_context = f"""
=== CRITICAL: INSPIRATION IMAGE ANALYSIS (HIGHEST PRIORITY) ===
The user uploaded an inspiration image. Here is the detailed analysis:

{inspiration_analysis}

=== MANDATORY REQUIREMENTS ===
1. Your generated prompt MUST incorporate the SUBJECT MATTER from the inspiration (if it shows a bear, your prompt must include a bear; if it shows a landscape, include that landscape, etc.)
2. Use the STYLE CUES, COLOR PALETTE, and LIGHTING described above
3. Incorporate the PROMPT SNIPPETS provided
4. Match the MOOD/ATMOSPHERE of the inspiration
5. The inspiration image takes PRIORITY over default BriteCo brand imagery - adapt the brand message to work WITH the inspiration, not against it

DO NOT ignore the inspiration image. DO NOT default to generic "happy couple with ring" if the inspiration shows something different.
"""

        # Build platform-specific context
        if is_google_ads:
            platform_context = f"""You are an expert at creating image generation prompts for Google Ads campaigns.

Create an image generation prompt for BriteCo jewelry insurance ads for {', '.join(platforms)}.

Campaign context: {campaign_text}
{inspiration_context}
{BRAND_GUIDELINES}

{GOOGLE_ADS_BEST_PRACTICES}

IMPORTANT FOR GOOGLE ADS:
- Images must be high-quality and work across YouTube, Discover, Gmail, and Display Network
- Subject should be centered in 80% of frame space
- Keep text overlay minimal (under 20% of image)
- Create aspirational lifestyle imagery that inspires action
- Ensure the imagery works well at smaller mobile sizes
- Use clear visual hierarchy with one strong focal point

NEVER include hex color codes like #31D7CA in your prompt - just describe the colors by name (turquoise, navy, orange).

Generate ONE detailed, creative prompt (200 words max) for Nano Banana (Google Gemini) image generator. Make it specific, visual, and actionable."""
        else:
            platform_context = f"""You are an expert at creating image generation prompts for AI models.

Create an image generation prompt for BriteCo jewelry insurance ads for {', '.join(platforms)}.

Campaign context: {campaign_text}
{inspiration_context}
{BRAND_GUIDELINES}

NEVER include hex color codes like #31D7CA in your prompt - just describe the colors by name (turquoise, navy, orange).

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
        elif provider == 'gemini':
            print("[API] Using Gemini...")
            result = gemini_client.generate_content(
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
        combined_brief = data.get('combinedBrief', '')  # NEW: Accept pre-merged brief
        platform = data.get('platform', 'tiktok')
        duration = data.get('duration', 6)
        motion_style = data.get('motionStyle', 'smooth')
        provider = data.get('provider', 'claude')

        print(f"\n[API] Generate Video Prompt Request")
        print(f"  Provider: {provider}")
        print(f"  Platform: {platform}")
        print(f"  Duration: {duration}s")
        print(f"  Motion: {motion_style}")
        print(f"  Combined brief length: {len(combined_brief)}")
        print(f"  Combined brief preview: {combined_brief[:300]}..." if combined_brief else "  WARNING: No combined brief!")

        # Camera motion descriptions
        motion_descriptions = {
            'smooth': 'smooth, fluid camera movement with gentle pans',
            'zoom': 'gradual zoom revealing details',
            'dynamic': 'dynamic camera angles with elegant movement'
        }
        motion_desc = motion_descriptions.get(motion_style, motion_descriptions['smooth'])

        # Use combined brief if available, otherwise fall back to campaign text
        brief_content = combined_brief if combined_brief else f"Campaign: {campaign_text}"

        prompt_context = f"""You are an expert at creating video generation prompts for AI video models like Google Veo.

Create a video generation prompt for a BriteCo jewelry insurance advertisement.

=== PRIMARY DIRECTIVE: USER'S CREATIVE BRIEF ===
{brief_content}

=== HOW TO COMBINE THE BRIEF WITH BRAND GUIDELINES ===
1. USE the subjects, scenes, and imagery described in the USER'S CREATIVE BRIEF above
2. APPLY the BriteCo brand style (colors, lighting, aesthetic) to those subjects
3. If the brief includes inspiration image analysis, incorporate those visual elements (subject matter, style, colors, lighting)
4. NEVER substitute the user's specified subjects with default imagery

=== BRITECO BRAND STYLE (Apply these to the user's subjects) ===
- Color accents: Turquoise, Navy, Orange (use visually in lighting, backgrounds, or props)
- Aesthetic: Modern, clean, optimistic, trustworthy
- Lighting: Warm, natural, professional quality
- No text overlays in the generated video

Platform: {platform}
Video Duration: {duration} seconds
Camera Style: {motion_desc}

VEO VIDEO BEST PRACTICES:
- Structure: Subject + Action + Style + Lighting
- Use clear, action-focused descriptions
- Avoid negative words like "no" or "don't"
- Describe camera movement explicitly
- Include lighting and atmosphere details
- Keep it under 500 words for best results

Create ONE detailed, cinematic video prompt that incorporates the user's creative brief with BriteCo's brand style.
The video should feel like a premium luxury brand commercial."""

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
    """Generate video using Google Veo 3.1 API via Vertex AI"""
    # Get OAuth2 token using ADC
    token = get_veo_auth_token()
    if not token:
        return jsonify({'success': False, 'error': 'Failed to authenticate with Google Cloud. Check service account permissions.'}), 500

    # Convert aspect ratio format (720:1280 -> 9:16)
    ratio_map = {
        '720:1280': '9:16',
        '1280:720': '16:9',
        '1080:1920': '9:16',
        '1920:1080': '16:9'
    }
    veo_aspect_ratio = ratio_map.get(aspect_ratio, '9:16')

    # Ensure duration is valid integer (5-8 for Veo 3.1)
    if duration < 5:
        duration = 5
    elif duration > 8:
        duration = 8

    # Vertex AI endpoint for Veo
    api_endpoint = f"https://{VEO_LOCATION}-aiplatform.googleapis.com"
    endpoint = f"{api_endpoint}/v1/projects/{VEO_PROJECT}/locations/{VEO_LOCATION}/publishers/google/models/{VEO_MODEL_ID}:predictLongRunning"

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    # Build payload for Vertex AI Veo API
    # Note: durationSeconds must be a string for Vertex AI
    payload = {
        'endpoint': f"projects/{VEO_PROJECT}/locations/{VEO_LOCATION}/publishers/google/models/{VEO_MODEL_ID}",
        'instances': [{
            'prompt': prompt_text
        }],
        'parameters': {
            'aspectRatio': veo_aspect_ratio,
            'durationSeconds': str(duration),
            'resolution': '720p',
            'sampleCount': 1,
            'personGeneration': 'allow_adult',
            'addWatermark': True,
            'includeRaiReason': True,
            'generateAudio': True,
            'enhancePrompt': True
        }
    }

    # Add image for image-to-video
    if prompt_image:
        # Extract base64 data and determine mime type
        if ',' in prompt_image:
            header, base64_data = prompt_image.split(',', 1)
            if 'png' in header.lower():
                mime_type = 'image/png'
            elif 'gif' in header.lower():
                mime_type = 'image/gif'
            else:
                mime_type = 'image/jpeg'
        else:
            base64_data = prompt_image
            mime_type = 'image/jpeg'

        payload['instances'][0]['image'] = {
            'bytesBase64Encoded': base64_data,
            'mimeType': mime_type
        }

    print(f"[API] Calling Google Veo API (Vertex AI): {endpoint}")
    print(f"[API] Project: {VEO_PROJECT}, Location: {VEO_LOCATION}, Model: {VEO_MODEL_ID}")
    print(f"[API] Duration: {duration}s, Aspect Ratio: {veo_aspect_ratio}")

    # Create the task
    response = requests.post(endpoint, headers=headers, json=payload, timeout=120)

    print(f"[API] Veo Response Status: {response.status_code}")

    if response.status_code != 200:
        error_msg = response.text
        print(f"[API ERROR] Veo API error: {error_msg}")
        return jsonify({'success': False, 'error': f'Veo API error: {error_msg}'}), 500

    task_data = response.json()
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
    """Check status of a Google Veo video generation task via Vertex AI"""
    # Get OAuth2 token using ADC
    token = get_veo_auth_token()
    if not token:
        return jsonify({'success': False, 'error': 'Failed to authenticate with Google Cloud'}), 500

    # Vertex AI polling endpoint
    api_endpoint = f"https://{VEO_LOCATION}-aiplatform.googleapis.com"
    endpoint = f"{api_endpoint}/v1/projects/{VEO_PROJECT}/locations/{VEO_LOCATION}/publishers/google/models/{VEO_MODEL_ID}:fetchPredictOperation"

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    payload = {
        'operationName': operation_name
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        return jsonify({'success': False, 'error': f'Failed to get Veo task status: {response.text}'}), 500

    task_data = response.json()
    done = task_data.get('done', False)
    status = task_data.get('status') or (task_data.get('response', {}).get('status')) or (task_data.get('metadata', {}).get('status'))

    result = {
        'success': True,
        'task_id': operation_name,
        'provider': 'veo'
    }

    if done or status in ('SUCCEEDED', 'COMPLETED', 'FINISHED'):
        # Check for error
        if 'error' in task_data:
            result['status'] = 'FAILED'
            result['error'] = task_data['error'].get('message', 'Unknown error')
            print(f"[API] Veo video generation failed: {result['error']}")
        else:
            result['status'] = 'SUCCEEDED'
            # Extract video URL from response
            response_data = task_data.get('response', {})

            # Try multiple paths to find the video
            videos = response_data.get('videos') or response_data.get('predictions') or []
            video_uri = None
            video_b64 = None

            if videos:
                v0 = videos[0]
                # Check for base64 data
                video_b64 = v0.get('bytesBase64Encoded') or (v0.get('inlineData', {}) or {}).get('data')
                # Check for URI
                for key in ('uri', 'videoUri', 'signedUri', 'downloadUri', 'fileUri', 'gcsUri'):
                    u = v0.get(key)
                    if isinstance(u, str) and u:
                        video_uri = u
                        break

            if video_b64:
                # Return as data URI
                result['video_url'] = f'data:video/mp4;base64,{video_b64}'
            elif video_uri:
                if video_uri.startswith('gs://'):
                    # GCS URI - need to proxy through our server
                    # Extract bucket and path
                    gcs_path = video_uri[5:]  # Remove gs://
                    result['video_url'] = f'/api/veo-video-gcs/{gcs_path}'
                    result['original_url'] = video_uri
                elif video_uri.startswith('http'):
                    # Direct HTTP URL (signed URL)
                    result['video_url'] = video_uri
                else:
                    result['video_url'] = video_uri

            print(f"[API] Veo video generation complete: {result.get('video_url', 'no URL')[:100]}...")
    elif status in ('FAILED', 'ERROR'):
        result['status'] = 'FAILED'
        result['error'] = task_data.get('error', {}).get('message', 'Video generation failed')
        print(f"[API] Veo video generation failed: {result['error']}")
    else:
        result['status'] = 'RUNNING'
        result['progress'] = 50  # Placeholder
        if status:
            print(f"[API] Veo video generation status: {status}")
        else:
            print(f"[API] Veo video generation in progress...")

    return jsonify(result)


@app.route('/api/veo-video-gcs/<path:gcs_path>')
def proxy_veo_video_gcs(gcs_path):
    """Proxy endpoint to serve Veo-generated videos from GCS"""
    try:
        from google.cloud import storage

        # Parse bucket and blob path
        parts = gcs_path.split('/', 1)
        if len(parts) != 2:
            return jsonify({'success': False, 'error': 'Invalid GCS path'}), 400

        bucket_name, blob_path = parts

        print(f"[API] Proxying GCS video: gs://{bucket_name}/{blob_path}")

        # Download from GCS
        storage_client = storage.Client(project=VEO_PROJECT)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Download to memory
        video_bytes = blob.download_as_bytes()

        print(f"[API] Downloaded {len(video_bytes)} bytes from GCS")

        return Response(
            video_bytes,
            content_type='video/mp4',
            headers={
                'Content-Disposition': f'inline; filename="veo_video.mp4"',
                'Cache-Control': 'public, max-age=3600'
            }
        )

    except Exception as e:
        print(f"[API ERROR] GCS video proxy error: {str(e)}")
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


@app.route('/api/trim-video', methods=['POST'])
def trim_video():
    """Trim video to specified start/end times and convert to MP4"""
    try:
        data = request.json
        video_url = data.get('video_url')
        start_time = float(data.get('start', 0))
        end_time = float(data.get('end'))

        if not video_url:
            return jsonify({'success': False, 'error': 'No video URL provided'}), 400

        if not end_time or end_time <= start_time:
            return jsonify({'success': False, 'error': 'Invalid trim times'}), 400

        duration = end_time - start_time
        print(f"[API] Trim video request: {start_time}s to {end_time}s ({duration}s)")

        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as input_file:
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as output_file:
            output_path = output_file.name

        try:
            # Handle data URL or HTTP URL
            if video_url.startswith('data:'):
                # Data URL - extract base64 data
                print(f"[API] Processing data URL...")
                # Format: data:video/mp4;base64,XXXXX
                if ';base64,' in video_url:
                    video_data = base64.b64decode(video_url.split(';base64,')[1])
                else:
                    return jsonify({'success': False, 'error': 'Invalid data URL format'}), 400

                with open(input_path, 'wb') as f:
                    f.write(video_data)
                print(f"[API] Data URL decoded, size: {len(video_data)} bytes")
            else:
                # HTTP URL - download
                print(f"[API] Downloading video from: {video_url[:100]}...")
                response = requests.get(video_url, stream=True, timeout=60)
                response.raise_for_status()

                with open(input_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            print(f"[API] Video ready, trimming with FFmpeg...")

            # Use FFmpeg to trim and convert to MP4
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-i', input_path,
                '-t', str(duration),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                print(f"[API] FFmpeg error: {result.stderr}")
                return jsonify({'success': False, 'error': f'FFmpeg error: {result.stderr[:500]}'}), 500

            print(f"[API] Video trimmed successfully")

            # Read the output file and return as base64
            with open(output_path, 'rb') as f:
                video_data = f.read()

            video_base64 = base64.b64encode(video_data).decode('utf-8')

            return jsonify({
                'success': True,
                'video_data': video_base64,
                'filename': f'trimmed_video_{start_time:.1f}s-{end_time:.1f}s.mp4',
                'duration': duration
            })

        finally:
            # Clean up temp files
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)

    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'FFmpeg timed out'}), 500
    except Exception as e:
        print(f"[API ERROR] Trim video error: {str(e)}")
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
    print(f"  [OK] Google Veo 3.1 (Vertex AI - uses ADC auth)")
    print(f"       Project: {VEO_PROJECT}, Location: {VEO_LOCATION}")
    print(f"  [OK] Runway (video generation - fallback)" if RUNWAY_API_KEY else "  [--] Runway (not configured)")
    print(f"\nPress Ctrl+C to stop")
    print("=" * 80)
    print()

    # Use PORT from environment (for Cloud Run) or default to 3000 for local dev
    app.run(debug=os.environ.get('FLASK_DEBUG', 'true').lower() == 'true', port=port, host='0.0.0.0')
