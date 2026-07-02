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
import secrets
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, Response, send_file, redirect, session, url_for
from flask_cors import CORS
from dotenv import load_dotenv
from authlib.integrations.flask_client import OAuth
from werkzeug.middleware.proxy_fix import ProxyFix

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

# Bucket Veo writes finished videos into (via storageUri), and the allowlist the
# GCS proxy is permitted to serve from. Prevents the proxy from reading arbitrary
# buckets the service account can access.
VEO_OUTPUT_BUCKET = os.getenv('VEO_OUTPUT_BUCKET', 'video-ad-generator-drafts')
ALLOWED_GCS_BUCKETS = {VEO_OUTPUT_BUCKET, 'video-ad-generator-drafts'}

# Cached ADC credentials for Vertex (avoids re-auth on every poll)
_veo_creds = None


def get_veo_auth_token():
    """Get OAuth2 access token for Vertex AI using Application Default Credentials.

    Credentials are cached at module scope and only refreshed when expired, so a
    5s status poll no longer hits the metadata server ~60x per video.
    """
    global _veo_creds
    try:
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        if _veo_creds is None:
            _veo_creds, project = google.auth.default(scopes=scopes)
            print(f"[VEO AUTH] Loaded ADC credentials for project: {project}", flush=True)
        # Refresh only when missing/expired (token is valid ~1 hour)
        if not _veo_creds.valid or _veo_creds.expired:
            print("[VEO AUTH] Refreshing credentials...", flush=True)
            _veo_creds.refresh(GoogleAuthRequest())
        return _veo_creds.token
    except Exception as e:
        import traceback
        print(f"[VEO AUTH ERROR] Failed to get ADC token: {e}", flush=True)
        print(f"[VEO AUTH ERROR] Traceback: {traceback.format_exc()}", flush=True)
        return None

# Import from local integrations folder
from integrations.openai_client import OpenAIClient
from integrations.gemini_client import GeminiClient
from integrations.claude_client import ClaudeClient

app = Flask(__name__, static_folder='.')
CORS(app)

# Fix for running behind Cloud Run's proxy - ensures correct HTTPS URLs
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Session configuration for OAuth
_secret = os.environ.get('FLASK_SECRET_KEY')
if not _secret:
    print("[WARNING] FLASK_SECRET_KEY not set - using a random per-process key. "
          "Sessions will not persist across instances/restarts. Set FLASK_SECRET_KEY in production.", flush=True)
    _secret = secrets.token_hex(32)
app.secret_key = _secret

# Harden session cookies (SECURE defaults on; set SESSION_COOKIE_SECURE=false for local http dev)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.environ.get('SESSION_COOKIE_SECURE', 'true').lower() == 'true',
)

# OAuth configuration
oauth = OAuth(app)
google_oauth = oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

ALLOWED_DOMAIN = 'brite.co'

def get_current_user():
    """Get current authenticated user from session"""
    return session.get('user')


# ========== API AUTH GUARD ==========
# Every /api/* route requires an authenticated @brite.co session, EXCEPT the
# endpoints below. Closes the previously wide-open API surface in one place.
API_AUTH_EXEMPT = {'/api/user'}  # returns its own {authenticated: ...} shape

@app.before_request
def _enforce_api_auth():
    path = request.path
    if not path.startswith('/api/'):
        return  # HTML page, OAuth, static assets handled elsewhere
    if request.method == 'OPTIONS':
        return  # allow CORS preflight
    if path in API_AUTH_EXEMPT:
        return
    user = session.get('user')
    email = (user or {}).get('email', '') if isinstance(user, dict) else ''
    if not user or not str(email).endswith(f'@{ALLOWED_DOMAIN}'):
        return jsonify({'success': False, 'error': 'Authentication required'}), 401


def safe_blob_name(name, allowed_prefixes):
    """Validate a user-supplied GCS blob name stays within allowed prefixes.
    Returns the name if safe, else None. Blocks traversal / prefix escape."""
    if not name or not isinstance(name, str):
        return None
    if '..' in name or name.startswith('/') or '\\' in name or '\x00' in name:
        return None
    if not any(name.startswith(p) for p in allowed_prefixes):
        return None
    return name


# ========== LIST INDEX SIDECARS ==========
# Each save writes a tiny '<name>.idx' sidecar with just the list-view fields, so
# list-drafts reads those instead of downloading every full draft (which may embed
# a whole video). '.idx' (not '.json') so older deployed code ignores these files.
def _index_blob_name(full_name):
    return (full_name[:-5] if full_name.endswith('.json') else full_name) + '.idx'

def _write_index(bucket, full_name, index_obj):
    try:
        bucket.blob(_index_blob_name(full_name)).upload_from_string(
            json.dumps(index_obj), content_type='application/json')
    except Exception as e:
        print(f"[INDEX WRITE WARN] {full_name}: {e}")

def _format_saved_at(iso_str):
    if not iso_str:
        return ''
    try:
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return dt.strftime('%b %d, %I:%M %p')
    except Exception:
        return iso_str

def _video_draft_index(blob_name, data):
    """Build the list-view index entry for a video draft (shape the frontend expects)."""
    return {
        'name': blob_name,
        '_sortKey': data.get('lastSavedAt', ''),
        'metadata': {
            'campaignName': data.get('campaignName'),
            'selectedPlatform': data.get('selectedPlatform'),
            'savedBy': data.get('lastSavedBy'),
            'savedAt': _format_saved_at(data.get('lastSavedAt', '')),
            'videoUrl': bool(data.get('videoUrl')),  # list only needs a has-video flag
        }
    }


# Initialize AI clients
openai_client = OpenAIClient()
gemini_client = GeminiClient()
claude_client = ClaudeClient()

print("[OK] OpenAI initialized")
print("[OK] Gemini initialized")
print("[OK] Claude initialized")

# GCS for drafts
GCS_BUCKET_NAME = 'video-ad-generator-drafts'
gcs_client = None
try:
    from google.cloud import storage as gcs_storage
    gcs_client = gcs_storage.Client()
    print("[OK] GCS initialized for drafts")
except Exception as e:
    print(f"[WARNING] GCS not available: {e}")

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
    """Serve the main HTML page with user info injected"""
    user = get_current_user()
    if not user:
        return redirect('/auth/login')

    with open('index.html', 'r', encoding='utf-8') as f:
        html = f.read()

    # Inject user info for server-side auth
    user_script = f'''<script>
    window.AUTH_USER = {json.dumps(user)};
    </script>
</head>'''
    html = html.replace('</head>', user_script, 1)

    return Response(html, mimetype='text/html')


# ============================================================================
# OAUTH AUTHENTICATION ROUTES
# ============================================================================

@app.route('/auth/login')
def auth_login():
    """Redirect to Google OAuth"""
    if get_current_user():
        return redirect('/')
    redirect_uri = url_for('auth_callback', _external=True)
    return google_oauth.authorize_redirect(redirect_uri)


@app.route('/auth/callback')
def auth_callback():
    """Handle OAuth callback from Google"""
    try:
        token = google_oauth.authorize_access_token()
        user_info = token.get('userinfo')

        if not user_info:
            return 'Failed to get user info', 400

        email = user_info.get('email', '')

        # Enforce domain restriction
        if not email.endswith(f'@{ALLOWED_DOMAIN}'):
            return f'''
            <html>
            <head><title>Access Denied</title></head>
            <body style="font-family: sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #272D3F;">
                <div style="text-align: center; color: white; padding: 2rem;">
                    <h1 style="color: #FC883A;">Access Denied</h1>
                    <p>Only @{ALLOWED_DOMAIN} email addresses are allowed.</p>
                    <p style="color: #A9C1CB;">You tried to sign in with: {email}</p>
                    <a href="/auth/login" style="color: #31D7CA;">Try again with a different account</a>
                </div>
            </body>
            </html>
            ''', 403

        # Store user in session
        session['user'] = {
            'email': email,
            'name': user_info.get('name', ''),
            'picture': user_info.get('picture', '')
        }

        return redirect('/')

    except Exception as e:
        print(f"[AUTH ERROR] OAuth callback failed: {e}")
        return f'Authentication failed: {str(e)}', 500


@app.route('/auth/logout')
def auth_logout():
    """Clear session and redirect to login"""
    session.pop('user', None)
    return redirect('/auth/login')

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

@app.route('/api/user')
def get_current_user_api():
    """Return current authenticated user info"""
    user = get_current_user()
    if not user:
        return jsonify({'authenticated': False}), 401
    return jsonify({'authenticated': True, 'user': user})

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

        # Build the full task list (platform x size x 2 variations)
        tasks = []
        for platform in platforms:
            platform_lower = platform.lower()
            sizes = PLATFORM_SIZES.get(platform_lower, PLATFORM_SIZES['meta'])
            for size in sizes:
                for variation_num in range(1, 3):  # 1, 2
                    tasks.append((platform, size, variation_num))

        def _render_image(task):
            """Generate + compress one image concurrently. Returns dict or None."""
            platform, size, variation_num = task
            try:
                print(f"[API] Generating {platform} - {size['name']} - Variation {variation_num}...")

                width = size['width']
                height = size['height']
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

                enhanced_prompt = f"{prompt}\n\nIMPORTANT: Compose this image specifically for {aspect_hint}. Use {composition_hint}. Frame: {width}x{height}px.\n\nDo NOT include any company logos, brand marks, watermarks, or text overlays in the image. Generate photography only without any branding elements."

                result = gemini_client.generate_image(
                    prompt=enhanced_prompt,
                    model="gemini-2.5-flash-image"
                )

                image_data = result.get('image_data', '')
                if not image_data:
                    print(f"[API] WARNING - No image data for {platform} {size['name']} - Variation {variation_num}")
                    return None

                try:
                    import base64
                    from PIL import Image, ImageOps
                    from io import BytesIO

                    image_bytes = base64.b64decode(image_data)
                    pil_image = Image.open(BytesIO(image_bytes))
                    pil_image = ImageOps.fit(pil_image, (width, height), Image.Resampling.LANCZOS, centering=(0.5, 0.5))
                    buffer = BytesIO()
                    pil_image.convert('RGB').save(buffer, format='JPEG', quality=85, optimize=True)
                    compressed_bytes = buffer.getvalue()
                    image_data = base64.b64encode(compressed_bytes).decode('utf-8')
                    print(f"[API] Final: {pil_image.size}, compressed to {len(compressed_bytes)} bytes")
                except Exception as resize_error:
                    print(f"[API] WARNING - Resize failed, using original: {resize_error}")

                print(f"[API] SUCCESS - {platform} {size['name']} - Variation {variation_num}")
                return {
                    'platform': platform,
                    'size': f"{size['name']} - Variation {variation_num}",
                    'width': width,
                    'height': height,
                    'url': f"data:image/jpeg;base64,{image_data}"
                }
            except Exception as img_error:
                print(f"[API ERROR] Failed to generate {platform} {size['name']} - Variation {variation_num}: {img_error}")
                return None

        # Generate images concurrently (I/O-bound). map() preserves task order.
        from concurrent.futures import ThreadPoolExecutor
        max_workers = int(os.environ.get('IMAGE_GEN_CONCURRENCY', '4'))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_render_image, tasks))
        images = [r for r in results if r]

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


@app.route('/api/refine-video-prompt', methods=['POST'])
def refine_video_prompt():
    """Merge a user's plain-language feedback into the existing video prompt.

    Veo can't edit an existing video, so conversational refinement works by
    revising the PROMPT (keeping what already works) and regenerating.
    """
    try:
        data = request.json
        current_prompt = data.get('currentPrompt', '')
        feedback = data.get('feedback', '')
        history = data.get('history', [])  # list of prior feedback strings
        provider = data.get('provider', 'claude')

        if not current_prompt or not feedback:
            return jsonify({'success': False, 'error': 'currentPrompt and feedback are required'}), 400

        print(f"\n[API] Refine Video Prompt Request")
        print(f"  Provider: {provider}")
        print(f"  Feedback: {feedback[:200]}")
        print(f"  Prior feedback rounds: {len(history)}")

        history_block = ""
        if history:
            history_block = "\n=== EARLIER CHANGES ALREADY APPLIED (keep these) ===\n" + \
                "\n".join(f"- {h}" for h in history if h)

        refine_context = f"""You are refining an existing AI video-generation prompt for a BriteCo jewelry insurance advertisement.

The user liked the previous video but wants specific changes. Your job is to produce a REVISED prompt that:
1. KEEPS everything the user did not ask to change (same subjects, scene, mood, BriteCo brand style)
2. APPLIES the user's requested change precisely
3. Stays a single, cohesive, cinematic Veo prompt (Subject + Action + Style + Lighting, under 500 words, no negative words like "no"/"don't", no on-screen text)

=== CURRENT PROMPT ===
{current_prompt}
{history_block}

=== USER'S REQUESTED CHANGE ===
{feedback}

Return ONLY the revised video prompt text — no preamble, no explanation, no quotes."""

        if provider == 'claude':
            result = claude_client.generate_content(
                prompt=refine_context,
                max_tokens=800,
                temperature=0.7
            )
            revised = result.get('content', '').strip()
        else:
            result = openai_client.generate_content(
                prompt=refine_context,
                max_tokens=800,
                temperature=0.7
            )
            revised = result.get('content', '').strip()

        print(f"[API] Refined video prompt generated ({len(revised)} chars)")

        return jsonify({
            'success': True,
            'prompt': revised,
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
            'enhancePrompt': True,
            # Write the finished MP4 to GCS instead of returning it inline as
            # base64. The status handler turns the returned gs:// URI into a
            # same-origin /api/veo-video-gcs proxy URL (keeps thumbnails working).
            'storageUri': f'gs://{VEO_OUTPUT_BUCKET}/veo-outputs/'
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

    response = requests.post(endpoint, headers=headers, json=payload, timeout=120)

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
    """Proxy Veo-generated videos from GCS.

    Locked to an allowlisted bucket (no arbitrary-object read), reuses the global
    client, streams in chunks (no whole-file-in-memory), and honors Range requests
    so HTML5 seeking / iOS Safari playback work.
    """
    try:
        import re as _re

        parts = gcs_path.split('/', 1)
        if len(parts) != 2:
            return jsonify({'success': False, 'error': 'Invalid GCS path'}), 400
        bucket_name, blob_path = parts

        # Only serve from allowlisted buckets, and block path traversal
        if bucket_name not in ALLOWED_GCS_BUCKETS or '..' in blob_path or blob_path.startswith('/'):
            return jsonify({'success': False, 'error': 'Forbidden'}), 403

        client = gcs_client or gcs_storage.Client(project=VEO_PROJECT)
        blob = client.bucket(bucket_name).blob(blob_path)
        if not blob.exists():
            return jsonify({'success': False, 'error': 'Not found'}), 404
        blob.reload()
        total = blob.size or 0

        range_header = request.headers.get('Range')
        if range_header and total:
            m = _re.match(r'bytes=(\d*)-(\d*)', range_header)
            start = int(m.group(1)) if m and m.group(1) else 0
            end = int(m.group(2)) if m and m.group(2) else total - 1
            end = min(end, total - 1)
            start = min(start, end)
            chunk = blob.download_as_bytes(start=start, end=end)  # end inclusive
            resp = Response(chunk, status=206, content_type='video/mp4')
            resp.headers['Content-Range'] = f'bytes {start}-{end}/{total}'
            resp.headers['Accept-Ranges'] = 'bytes'
            resp.headers['Content-Length'] = str(len(chunk))
            resp.headers['Cache-Control'] = 'private, max-age=3600'
            return resp

        # Full file: stream in chunks so we never hold the whole video in memory
        def _generate():
            with blob.open('rb') as f:
                while True:
                    data = f.read(262144)  # 256 KB
                    if not data:
                        break
                    yield data

        resp = Response(_generate(), content_type='video/mp4')
        resp.headers['Accept-Ranges'] = 'bytes'
        if total:
            resp.headers['Content-Length'] = str(total)
        resp.headers['Content-Disposition'] = 'inline; filename="veo_video.mp4"'
        resp.headers['Cache-Control'] = 'private, max-age=3600'
        return resp

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

    response = requests.get(f'{RUNWAY_API_BASE}/tasks/{task_id}', headers=headers, timeout=30)

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


def _resolve_proxy_video_blob(video_url):
    """Resolve a trim source to a GCS blob, ONLY if it's our own veo-video-gcs
    proxy path pointing at an allowlisted bucket. Returns a blob or None.
    Prevents the server from fetching arbitrary attacker-supplied URLs (SSRF)."""
    from urllib.parse import unquote
    marker = '/api/veo-video-gcs/'
    if not video_url or marker not in video_url:
        return None
    gcs_path = unquote(video_url.split(marker, 1)[1].split('?', 1)[0])
    parts = gcs_path.split('/', 1)
    if len(parts) != 2:
        return None
    bucket_name, blob_path = parts
    if bucket_name not in ALLOWED_GCS_BUCKETS or '..' in blob_path or blob_path.startswith('/'):
        return None
    try:
        client = gcs_client or gcs_storage.Client(project=VEO_PROJECT)
        return client.bucket(bucket_name).blob(blob_path)
    except Exception:
        return None


@app.route('/api/trim-video', methods=['POST'])
def trim_video():
    """Trim video to specified start/end times, apply speed, and convert to MP4"""
    try:
        data = request.json
        video_url = data.get('video_url')
        start_time = float(data.get('start', 0))
        end_time = float(data.get('end'))
        speed = float(data.get('speed', 1.0))

        if not video_url:
            return jsonify({'success': False, 'error': 'No video URL provided'}), 400

        if not end_time or end_time <= start_time:
            return jsonify({'success': False, 'error': 'Invalid trim times'}), 400

        # Clamp speed to reasonable range
        speed = max(0.5, min(4.0, speed))

        duration = end_time - start_time
        final_duration = duration / speed  # Actual duration after speed change
        print(f"[API] Trim video request: {start_time}s to {end_time}s ({duration}s) at {speed}x speed -> {final_duration:.2f}s final")

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
                # Only our own GCS proxy path is allowed; read straight from GCS
                # (no arbitrary server-side URL fetch -> no SSRF).
                blob = _resolve_proxy_video_blob(video_url)
                if blob is None:
                    return jsonify({'success': False, 'error': 'Unsupported video source. Only generated videos can be trimmed.'}), 400
                print(f"[API] Downloading video from GCS proxy source...")
                blob.download_to_filename(input_path)

            print(f"[API] Video ready, trimming with FFmpeg...")

            # Build FFmpeg command with optional speed adjustment
            if speed != 1.0:
                # Apply speed change: setpts for video, atempo for audio
                # setpts=PTS/speed speeds up (2x = PTS/2), slows down (0.5x = PTS/0.5)
                video_filter = f"setpts=PTS/{speed}"
                # atempo only supports 0.5-2.0, chain for higher speeds
                if speed <= 2.0:
                    audio_filter = f"atempo={speed}"
                else:
                    # Chain atempo filters for speeds > 2x (e.g., 4x = atempo=2,atempo=2)
                    audio_filter = f"atempo=2,atempo={speed/2}"

                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start_time),
                    '-i', input_path,
                    '-t', str(duration),
                    '-vf', video_filter,
                    '-af', audio_filter,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-movflags', '+faststart',
                    output_path
                ]
            else:
                # No speed change, just trim
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
                'filename': f'trimmed_video_{start_time:.1f}s-{end_time:.1f}s_{speed}x.mp4',
                'duration': final_duration,
                'original_duration': duration,
                'speed': speed
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


# ============================================================================
# DRAFTS - GCS Auto-save
# ============================================================================

@app.route('/api/save-draft', methods=['POST'])
def save_draft():
    """Save video ad project draft to GCS"""
    if not gcs_client:
        return jsonify({'success': False, 'error': 'GCS not available'}), 503
    try:
        data = request.json
        campaign_name = data.get('campaignName', 'untitled').lower().replace(' ', '-')[:50]
        saved_by = data.get('savedBy', 'unknown').split('@')[0].replace('.', '-')
        blob_name = f"drafts/{campaign_name}-{saved_by}.json"

        draft = {
            'campaignName': data.get('campaignName', 'Untitled'),
            'currentStep': data.get('currentStep'),
            'selectedPlatform': data.get('selectedPlatform'),
            'campaignText': data.get('campaignText'),
            'adCopy': data.get('adCopy'),
            'videoUrl': data.get('videoUrl'),
            'videoPrompt': data.get('videoPrompt'),
            'lastSavedBy': data.get('savedBy', 'unknown'),
            'lastSavedAt': datetime.now().isoformat()
        }

        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(draft), content_type='application/json')
        _write_index(bucket, blob_name, _video_draft_index(blob_name, draft))
        return jsonify({'success': True, 'file': blob_name})

    except Exception as e:
        print(f"[DRAFT SAVE ERROR] {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/list-drafts', methods=['GET'])
def list_drafts():
    """List all drafts from GCS"""
    if not gcs_client:
        return jsonify([])
    try:
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix='drafts/'))
        have_idx = {b.name for b in blobs if b.name.endswith('.idx')}
        drafts = []
        for blob in blobs:
            if blob.name.endswith('.idx'):
                try:
                    drafts.append(json.loads(blob.download_as_text()))
                except Exception as idx_err:
                    print(f"[DRAFT INDEX READ WARN] {blob.name}: {idx_err}")
            elif blob.name.endswith('.json'):
                # Old item with no sidecar yet: download once, then self-heal.
                if _index_blob_name(blob.name) in have_idx:
                    continue
                data = json.loads(blob.download_as_text())
                entry = _video_draft_index(blob.name, data)
                drafts.append(entry)
                _write_index(bucket, blob.name, entry)
        drafts.sort(key=lambda d: d.get('_sortKey', '') or d.get('metadata', {}).get('savedAt', ''), reverse=True)
        return jsonify(drafts)
    except Exception as e:
        print(f"[DRAFT LIST ERROR] {str(e)}")
        return jsonify([])


@app.route('/api/load-draft', methods=['GET'])
def load_draft():
    """Load a specific draft from GCS"""
    if not gcs_client:
        return jsonify({'success': False, 'error': 'GCS not available'}), 503
    try:
        filename = safe_blob_name(request.args.get('file'), ('drafts/',))
        if not filename:
            return jsonify({'success': False, 'error': 'Invalid or missing file'}), 400
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(filename)
        if not blob.exists():
            return jsonify({'success': False, 'error': 'Draft not found'}), 404
        data = json.loads(blob.download_as_text())
        # Return draft data directly for frontend compatibility
        return jsonify(data)
    except Exception as e:
        print(f"[DRAFT LOAD ERROR] {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/delete-draft', methods=['DELETE'])
def delete_draft():
    """Delete a draft from GCS"""
    if not gcs_client:
        return jsonify({'success': True})
    try:
        filename = safe_blob_name((request.json or {}).get('file'), ('drafts/',))
        if not filename:
            return jsonify({'success': False, 'error': 'Invalid or missing file'}), 400
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(filename)
        if blob.exists():
            blob.delete()
        idx = bucket.blob(_index_blob_name(filename))
        if idx.exists():
            idx.delete()
        return jsonify({'success': True})
    except Exception as e:
        print(f"[DRAFT DELETE ERROR] {str(e)}")
        return jsonify({'success': False, 'error': 'Delete failed'}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 3000))

    print("=" * 80)
    print("BriteCo Ad Generator - Python API Server")
    print("=" * 80)
    print(f"\nStarting server...")
    print(f"URL: http://localhost:{port}")
    print(f"\nAPIs:")
    print(f"  [OK] Claude (claude-sonnet-4-6)")
    print(f"  [OK] OpenAI (gpt-5.5)")
    print(f"  [OK] Google Gemini (gemini-2.5-flash-image / Nano Banana)")
    print(f"  [OK] Google Veo 3.1 (Vertex AI - uses ADC auth)")
    print(f"       Project: {VEO_PROJECT}, Location: {VEO_LOCATION}")
    print(f"  [OK] Runway (video generation - fallback)" if RUNWAY_API_KEY else "  [--] Runway (not configured)")
    print(f"\nPress Ctrl+C to stop")
    print("=" * 80)
    print()

    # Use PORT from environment (for Cloud Run) or default to 3000 for local dev
    app.run(debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true', port=port, host='0.0.0.0')
