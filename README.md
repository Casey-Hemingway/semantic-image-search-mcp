# Semantic Image Search MCP Server

Search your photo archive using natural language with AI-powered semantic understanding. Built as an MCP (Model Context Protocol) server for seamless integration with Claude Desktop.

## Features

- **Semantic Search**: Find images by describing what's in them, not just filenames
- **Zero Configuration**: No manual tagging required - works out of the box
- **EXIF Metadata**: Automatically extracts camera settings, dates, and GPS data
- **Fast Indexing**: Optimized for Apple Silicon (MPS) and NVIDIA GPUs (CUDA)
- **Claude Integration**: Works natively with Claude Desktop via MCP
- **Privacy First**: Runs 100% locally - your photos never leave your machine

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/himalayantrust/semantic-image-search-mcp
cd semantic-image-search-mcp

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example configuration
cp config.yml.example config.yml

# Edit config.yml with your photo archive path
nano config.yml  # or use your preferred editor
```

Minimal configuration:
```yaml
archive_path: "/path/to/your/photos"
```

### 3. Index Your Photos

```bash
# Run initial indexing
python3 -c "
import asyncio
from pathlib import Path
from src.config import Config
from src.indexer import ImageIndexer

async def index():
    config = Config.from_yaml(Path('config.yml'))
    indexer = ImageIndexer(config)
    stats = await indexer.index_archive()
    print(f'Indexed {stats[\"indexed\"]} images')

asyncio.run(index())
"
```

### 4. Set Up Claude Desktop Integration

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "semantic-image-search": {
      "command": "python3",
      "args": ["/absolute/path/to/photo-library/run_server.py"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/photo-library"
      }
    }
  }
}
```

### 5. Restart Claude Desktop

After updating the configuration, restart Claude Desktop. You should see the `semantic-image-search` server connected in the MCP section.

## Usage Examples

### Search for Images

Ask Claude:
```
Search my photos for images with people in classrooms
```

```
Find photos of mountain landscapes taken in 2024
```

```
Show me portraits with natural lighting
```

### Get Image Details

```
Get detailed information about image abc123def456
```

### View Archive Statistics

```
Show me statistics about my photo archive
```

### Reindex After Adding Photos

```
Reindex my photo archive
```

## MCP Tools

The server exposes four tools to Claude:

### 1. `search_images`
Search images using natural language queries with optional filters.

**Parameters:**
- `query` (string, required): Natural language description
- `limit` (integer, optional): Max results (default: 10, max: 100)
- `date_from` (string, optional): Filter by date (ISO format: YYYY-MM-DD)
- `date_to` (string, optional): Filter by date (ISO format: YYYY-MM-DD)
- `folder_pattern` (string, optional): Filter by folder path

**Example:**
```python
{
  "query": "person standing in a room",
  "limit": 5,
  "date_from": "2024-01-01"
}
```

### 2. `get_image_info`
Get detailed metadata for a specific image.

**Parameters:**
- `image_id` (string, required): Unique image identifier

### 3. `reindex_archive`
Re-index the photo archive for new or modified images.

**Parameters:**
- `force` (boolean, optional): Force re-index all images (default: false)

### 4. `get_archive_stats`
Get statistics about the indexed photo archive.

**No parameters required.**

## Configuration Reference

```yaml
# Path to your photo archive (required)
archive_path: "/path/to/photos"

# Directory for storing index data (optional)
data_dir: "./data"

# CLIP model configuration
clip:
  # Model to use for embeddings
  model_name: "openai/clip-vit-base-patch32"  # or "openai/clip-vit-large-patch14"

  # Device for inference
  device: "auto"  # auto, mps, cuda, or cpu

  # Batch size for processing
  batch_size: 32  # Increase for more RAM/VRAM

# Search configuration
search:
  default_limit: 10
  max_limit: 100
  similarity_threshold: 0.0  # 0.0 = show all ranked results

# Thumbnail configuration
thumbnails:
  enabled: true
  max_size: 512
  quality: 85
```

## Architecture

### Technology Stack

- **CLIP**: OpenAI's vision-language model for understanding images
- **FAISS**: Facebook's vector similarity search library
- **SQLite**: Lightweight database for metadata storage
- **MCP**: Model Context Protocol for Claude integration
- **PyTorch**: ML framework with Apple Silicon (MPS) support

### How It Works

1. **Indexing**:
   - Scans your archive for image files
   - Extracts EXIF metadata (camera, date, location, etc.)
   - Generates semantic embeddings using CLIP
   - Stores embeddings in FAISS vector index
   - Saves metadata in SQLite database

2. **Searching**:
   - Converts your text query to an embedding
   - Searches FAISS index for similar image embeddings
   - Applies filters (date, folder, etc.)
   - Returns ranked results with similarity scores

3. **MCP Integration**:
   - Exposes search tools to Claude via stdio protocol
   - Claude can search, get details, and manage your archive
   - All processing happens locally on your machine

## Performance

### Indexing Speed (Apple Silicon M-series)

- **Small archives** (< 1,000 images): ~30 seconds
- **Medium archives** (1,000 - 10,000 images): 2-5 minutes
- **Large archives** (10,000+ images): 10-30 minutes

### Search Latency

- **Typical query**: 200-500ms
- **With filters**: 300-700ms

### Memory Usage

- **Base**: ~200MB (model + server)
- **Per 10,000 images**: ~20MB (embeddings + metadata)

## Troubleshooting

### "FAISS index not found" Error

Run indexing first:
```bash
python3 -c "import asyncio; from src.indexer import ImageIndexer; from src.config import Config; from pathlib import Path; asyncio.run(ImageIndexer(Config.from_yaml(Path('config.yml'))).index_archive())"
```

### MCP Server Not Connecting

1. Check Claude Desktop logs: `~/Library/Logs/Claude/mcp*.log`
2. Verify absolute paths in `claude_desktop_config.json`
3. Ensure `config.yml` exists in the project directory
4. Check `mcp-server.log` for errors

### Slow Indexing

- Reduce `batch_size` in config.yml (uses less memory, slightly slower)
- Check that MPS/CUDA is being used (look for "Using Apple Silicon MPS" message)
- Close other applications to free up RAM

### Import Errors

Ensure virtual environment is activated:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
ruff check src/
```

## Use Cases

### Museums & Archives
Search historical photo collections by content, era, or subject matter.

### NGOs & Field Work
Find photos from specific trips, locations, or events for reports and social media.

### Media Companies
Quickly locate stock footage and images matching creative briefs.

### Photographers
Organize and search large portfolio collections by visual content.

### Researchers
Find specific images in large datasets for analysis and publication.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on [CLIP](https://github.com/openai/CLIP) by OpenAI
- Uses [FAISS](https://github.com/facebookresearch/faiss) by Meta AI Research
- Implements [Model Context Protocol](https://modelcontextprotocol.io/) by Anthropic

## Support

For issues and questions:
- GitHub Issues: https://github.com/himalayantrust/semantic-image-search-mcp/issues
- Email: info@himalayantrust.org

---

**Built with love by the Himalayan Trust team** 🏔️
