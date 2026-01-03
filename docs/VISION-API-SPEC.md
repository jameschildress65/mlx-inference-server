# MLX Inference Server - Vision API Specification
**For Integration with External Systems**
**Version:** Current (vision support enabled)
**Last Updated:** 2025-12-30

---

## Overview

MLX Inference Server provides OpenAI-compatible vision inference using Apple Silicon optimized models (MLX framework).

**What This Server Provides:**
- Vision-language model inference (multimodal)
- Base64-encoded image processing
- OpenAI-compatible API format
- High performance on Apple Silicon (~54-95 tok/s)

**What This Server Does NOT Provide:**
- PDF conversion (client responsibility)
- OCR (use vision model or external OCR)
- Image preprocessing (resize, compress, etc.)
- File storage or management

---

## API Endpoint

```
POST http://localhost:11440/v1/chat/completions
```

**Protocol:** HTTP (local only, no TLS required)
**Content-Type:** application/json

---

## Request Format

### Multimodal Content (Vision)

```json
{
  "model": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Analyze this scorecard and extract the leadership scores."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
          }
        }
      ]
    }
  ],
  "max_tokens": 500,
  "temperature": 0.7
}
```

### Text-Only Content

```json
{
  "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
  "messages": [
    {
      "role": "user",
      "content": "Explain quantum computing in simple terms."
    }
  ],
  "max_tokens": 200
}
```

---

## Parameters

### Required
- **model** (string): Model ID to use
  - Vision: `mlx-community/Qwen2.5-VL-7B-Instruct-4bit`
  - Text: Any supported MLX text model
- **messages** (array): Conversation messages

### Optional
- **max_tokens** (integer): Maximum tokens to generate (default: model-dependent)
- **temperature** (float): Sampling temperature 0.0-2.0 (default: 0.7)
- **top_p** (float): Nucleus sampling threshold (default: 1.0)
- **stream** (boolean): Stream response (not currently supported)

---

## Image Format Requirements

### Encoding
**Base64 data URI format:**
```
data:image/{format};base64,{base64_encoded_data}
```

**Supported formats:**
- `image/png`
- `image/jpeg`
- `image/webp`

### Size Limits
**Hard limit:** 10 MB per image (cannot override - security hardening)

**Consequences of exceeding:**
```json
{
  "error": {
    "message": "Request entity too large",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

### Dimension Limits (v3.1.0+)

**Automatic resizing:** Images are automatically resized based on available system RAM to prevent memory exhaustion.

**Why this matters:**
- Vision models use quadratic memory scaling (patches² × layers)
- Large images can cause 90-100GB+ memory allocation
- Example: 2550×3300px image → 90-100GB RAM usage

**Auto-resize limits by system RAM:**

| System RAM | Max Dimension | Typical Systems |
|------------|---------------|-----------------|
| 128+ GB | 1024px | Mac Studio, high-end workstations |
| 64 GB | 1024px | Mac Pro, professional systems |
| 32 GB | 768px | MacBook Air M4, standard laptops |
| 16 GB | 512px | Entry-level M-series Macs |
| 8 GB | 512px | Minimum configuration |

**Resize behavior:**
- Preserves aspect ratio (high-quality Lanczos resampling)
- Transparent to API clients (no special handling needed)
- Images cached with TTL to avoid redundant processing
- Happens automatically before inference

**Configuration (optional):**
```bash
# Override auto-detection
export VISION_MAX_DIMENSION=1024

# Disable auto-resize (not recommended - may cause OOM)
export VISION_AUTO_RESIZE=false

# Cache configuration
export VISION_RESIZE_CACHE_SIZE=20      # Number of cached images
export VISION_RESIZE_CACHE_TTL=3600     # TTL in seconds (1 hour)
```

**Best practice:** Let the server auto-configure based on available RAM. Manual override only needed for specific use cases.

### Encoding Example (Python)

```python
import base64

# Read image file
with open("image.png", "rb") as f:
    image_data = f.read()

# Encode to base64
image_b64 = base64.b64encode(image_data).decode('utf-8')

# Create data URI
data_uri = f"data:image/png;base64,{image_b64}"

# Use in request
content = [
    {"type": "text", "text": "What do you see?"},
    {"type": "image_url", "image_url": {"url": data_uri}}
]
```

---

## Response Format

### Success Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1735567890,
  "model": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "This scorecard shows a leadership assessment with color-coded ratings. The red items indicate areas requiring significant improvement: Technical Savvy, Character, Passionate Team Builder, and Influence & Persuasion..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 85,
    "total_tokens": 235,
    "tokens_per_sec": 54.25
  }
}
```

### Response Fields

- **id** (string): Unique completion ID
- **choices** (array): Generated responses (always 1 element)
  - **message.content** (string): Generated text
  - **finish_reason** (string): "stop" (completed) or "length" (max_tokens hit)
- **usage** (object): Token statistics
  - **tokens_per_sec** (float): Performance metric (MLX-specific extension)

---

## Error Responses

### 400 Bad Request - Invalid Input

```json
{
  "error": {
    "message": "Invalid base64 encoding in image_url",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

**Common causes:**
- Malformed base64 data
- Image exceeds 10MB
- Invalid data URI format
- Missing required fields

### 500 Internal Server Error - Server Error

```json
{
  "error": {
    "message": "Worker error: GenerationResult processing failed",
    "type": "server_error",
    "code": 500
  }
}
```

**Common causes:**
- Model crash
- Out of memory
- Worker process died
- Invalid model configuration

### 503 Service Unavailable - Server Not Ready

```json
{
  "error": {
    "message": "Worker not ready",
    "type": "service_unavailable_error",
    "code": 503
  }
}
```

**Cause:** Server starting up or worker crashed

**Fix:** Wait 5-10 seconds, retry, or restart server

---

## Performance Characteristics

### Vision Model (Qwen2.5-VL-7B-Instruct-4bit)
- **Speed:** ~54-95 tokens/second
- **Memory:** ~5.8 GB (base model) + image processing overhead
  - Small images (512px): ~5.8 GB total
  - Medium images (768px): ~5.8-6 GB total
  - Large images (1024px): ~6-8 GB total
  - Very large images: Automatically resized to safe limits (v3.1.0+)
- **First request:** Slower (model loading ~5-10s)
- **Subsequent:** Fast (model cached)

### Text Model (Qwen2.5-0.5B-Instruct-4bit)
- **Speed:** ~71-85 tokens/second
- **Memory:** ~400 MB
- **First request:** Fast (small model)

### Concurrency
- **Workers:** 1 (single process)
- **Concurrent requests:** 1 at a time (sequential processing)
- **Queue:** Requests wait in line

---

## Multi-Page Document Strategy

### Server Does NOT Support
- ❌ Multiple images in single message (depends on model)
- ❌ Document synthesis across pages
- ❌ Page-aware processing

### Client Responsibility
Choose one of these approaches:

#### Option 1: Sequential Processing
```python
for page_num, page_image in enumerate(pdf_images):
    response = send_to_mlx(page_image, f"Analyze page {page_num+1}")
    page_results.append(response)

# Synthesize results client-side
final_analysis = synthesize_pages(page_results)
```

#### Option 2: First Page Only
```python
first_page = pdf_images[0]
response = send_to_mlx(first_page, "Analyze this document")
```

#### Option 3: Multi-Image Message (Model Dependent)
```python
# May not work with all models
content = [{"type": "text", "text": "Analyze all pages"}]
for img in pdf_images:
    content.append({"type": "image_url", "image_url": {"url": img_b64}})

response = send_to_mlx(content)
```

**Recommendation:** Option 1 (sequential) for reliability

---

## PDF Conversion (Client-Side)

### Required
Clients must convert PDF → images before sending to MLX server.

### Recommended Libraries (Python)
```bash
pip install pdf2image PyPDF2
brew install poppler  # macOS (required for pdf2image)
```

### DPI Recommendations

| DPI | File Size (per page) | Quality | Use Case |
|-----|---------------------|---------|----------|
| 100 | 0.15-0.40 MB | Lower | Fast processing, simple docs |
| 150 | 0.30-0.80 MB | Good | Balanced (recommended) |
| 200 | 0.50-1.50 MB | High | Complex graphics, small text |
| 300 | 1.00-3.00 MB | Very high | Risk of exceeding 10MB limit |

**Safe default:** DPI 150, PNG format

### Format Comparison (same DPI)
- **PNG:** Lossless, larger files, best quality
- **JPEG 85%:** Lossy, ~50-70% size of PNG, good quality
- **JPEG 70%:** Lossy, ~40-50% size of PNG, acceptable quality

### Size Validation
**Always validate before sending:**

```python
def check_size(image_bytes: bytes, max_mb: int = 10) -> bool:
    size_mb = len(image_bytes) / (1024 * 1024)
    return size_mb <= max_mb

if not check_size(image_data):
    # Reduce DPI or use JPEG compression
    raise ValueError(f"Image exceeds {max_mb}MB limit")
```

---

## Server Management

### Health Check
```bash
curl http://localhost:11440/health
```

**Expected response:**
```json
{"status": "healthy"}
```

### Start Server
```bash
cd ~/projects/mlx-inference-server
./bin/mlx-inference-server-daemon.sh start
```

### Stop Server
```bash
./bin/mlx-inference-server-daemon.sh stop
```

### Status Check
```bash
./bin/mlx-inference-server-daemon.sh status
```

### Logs
```bash
tail -f logs/server.log
```

---

## Integration Checklist

### Before First Request
- [ ] Server is running (`./bin/mlx-inference-server-daemon.sh status`)
- [ ] Health check passes (`curl http://localhost:11440/health`)
- [ ] Vision model loaded (Qwen2.5-VL-7B-Instruct-4bit)

### Client Implementation
- [ ] PDF → image conversion (pdf2image)
- [ ] Base64 encoding (standard library)
- [ ] Size validation (<10MB)
- [ ] Error handling (400, 500, 503)
- [ ] Timeout handling (recommend 60s)
- [ ] Retry logic (for 503 errors)

### Testing
- [ ] Text-only request works
- [ ] Vision request with small image works
- [ ] Vision request with large image (near 10MB) works
- [ ] Error handling for >10MB image works
- [ ] Multi-page strategy tested

---

## Example: Complete PDF Analysis (Python)

```python
import requests
import base64
from pdf2image import convert_from_path

def analyze_pdf(pdf_path: str, prompt: str, dpi: int = 150) -> str:
    """
    Analyze PDF using MLX vision server.

    Args:
        pdf_path: Path to PDF file
        prompt: Analysis prompt
        dpi: Image resolution (default 150)

    Returns:
        Vision analysis text
    """
    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=dpi)

    # Use first page (Phase 1)
    first_page = images[0]

    # Convert to bytes
    import io
    buf = io.BytesIO()
    first_page.save(buf, format='PNG')
    img_bytes = buf.getvalue()

    # Validate size
    size_mb = len(img_bytes) / (1024 * 1024)
    if size_mb > 10:
        raise ValueError(f"Image {size_mb:.2f}MB exceeds 10MB limit. Try lower DPI.")

    # Encode to base64
    img_b64 = base64.b64encode(img_bytes).decode()
    data_uri = f"data:image/png;base64,{img_b64}"

    # Send to MLX server
    response = requests.post(
        "http://localhost:11440/v1/chat/completions",
        json={
            "model": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }],
            "max_tokens": 500
        },
        timeout=60
    )

    # Handle response
    if response.status_code == 200:
        data = response.json()
        return data['choices'][0]['message']['content']
    else:
        raise RuntimeError(f"MLX request failed: {response.status_code} - {response.text}")

# Usage
result = analyze_pdf(
    "/path/to/document.pdf",
    "Analyze this scorecard and extract key findings."
)
print(result)
```

---

## Limitations

### Cannot Do
- ❌ Convert PDFs (client must do this)
- ❌ Accept images >10MB (security limit)
- ❌ Process multiple concurrent requests (single worker)
- ❌ Stream responses (not implemented)
- ❌ Store images long-term (stateless, but has short-term TTL cache)
- ❌ Image enhancement or filtering

### Can Do
- ✅ Process base64-encoded images
- ✅ Vision-language inference
- ✅ OpenAI-compatible API
- ✅ High performance on Apple Silicon
- ✅ Multiple model support (text and vision)
- ✅ **Automatic image resizing** (v3.1.0+) based on system RAM
- ✅ **Resize caching** (v3.1.0+) with TTL for performance

---

## Security Considerations

### Image Size Limit
**10MB limit is non-negotiable** (security hardening from Opus 4.5 review).

**Rationale:**
- Prevents DoS attacks
- Prevents memory exhaustion
- Forces reasonable image sizes
- Encourages efficient preprocessing

### Local Only
Server listens on `localhost:11440` only (not exposed to network).

**Do NOT:**
- Expose to public internet
- Accept untrusted images without validation
- Bypass size limits

---

## Support

### Documentation
- **Vision Setup:** `docs/VISION-SETUP.md`
- **Main README:** `README.md`
- **Unauthorized Changes:** `docs/UNAUTHORIZED-RAGMT-CHANGES.md`

### Logs
```bash
tail -f logs/server.log
```

### Issues
Check git repo issues or server logs for troubleshooting.

---

## Summary for Integrators

### What You Need to Know
1. **Server provides:** Vision inference via OpenAI-compatible API
2. **Server does NOT provide:** PDF conversion (you do this)
3. **Hard limit:** 10MB per image (validate client-side)
4. **Format:** Base64 data URI
5. **Performance:** ~54-95 tok/s (vision), ~71-85 tok/s (text)
6. **Concurrency:** Sequential (1 request at a time)

### What You Need to Implement
1. PDF → image conversion (pdf2image)
2. Base64 encoding
3. Size validation (<10MB)
4. Error handling
5. Multi-page strategy (if needed)

### Quick Start
1. Start server: `./bin/mlx-inference-server-daemon.sh start`
2. Health check: `curl http://localhost:11440/health`
3. Send request: See example code above
4. Handle response: Extract `choices[0].message.content`

---

**Version:** Current (2025-12-30)
**Maintained by:** MLX Inference Server
**For questions:** See server documentation or logs
