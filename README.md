# BriteCo Video Ad Generator

AI-powered video advertisement generator for BriteCo jewelry insurance marketing.

## Features

- **AI Video Generation**: Create engaging video ads using AI
- **Multiple Templates**: Various video ad formats and styles
- **Brand Consistency**: Pre-loaded BriteCo logos and brand colors
- **Real-time Preview**: See your video ads before exporting

## Setup

1. Clone the repository
2. Copy `.env.example` to `.env`
3. Add your API keys to `.env`:
   ```
   ANTHROPIC_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   GOOGLE_AI_API_KEY=your_key_here
   ```
4. Install Python dependencies: `pip install flask flask-cors requests`
5. Run the server: `python ad_api_server.py`
6. Open `index.html` in your browser

## Files

- `index.html` - Video ad generator UI
- `ad_api_server.py` - Python backend server for AI integrations
- `.env.example` - Template for environment variables

## API Keys Required

- **Anthropic (Claude)** - For AI text generation
- **OpenAI** - Alternative AI text generation
- **Google AI (Gemini)** - For image generation
