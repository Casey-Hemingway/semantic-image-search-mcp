# Quick Start Guide

## Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

**Note**: First installation will download the CLIP model (~150MB) and may take a few minutes.

## Step 2: Test Indexing

The project includes 7 sample images from "SK Monitoring 2024" trips. Test the indexing with these:

```bash
python3 test_indexing.py
```

This will:
1. Load your configuration
2. Index the sample images
3. Test search queries
4. Display archive statistics

Expected output:
```
Semantic Image Search - Test Indexing
============================================================

1. Loading configuration...
   ✓ Configuration loaded
   Archive path: /Users/casey-hemingway/Documents/Projects/HT/photo-library

2. Indexing images...
Loading CLIP model: openai/clip-vit-base-patch32...
✓ Using Apple Silicon MPS (Metal Performance Shaders)
✓ CLIP model loaded successfully
Scanning archive: /Users/casey-hemingway/Documents/Projects/HT/photo-library
Found 7 images
Indexing images (batch size: 32)...
Building FAISS index...
✓ FAISS index saved: 7 vectors, dimension 512

   ✓ Indexing complete!
   Total images found: 7
   Successfully indexed: 7
   Errors: 0
   Duration: ~30s

3. Testing search functionality...
   Query: 'person in a room'
   Found 3 results:
      1. SK Monitoring 2024 (22.04.24) -1466.jpg (similarity: 0.892)
      ...
```

## Step 3: Configure Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "semantic-image-search": {
      "command": "python3",
      "args": ["/Users/casey-hemingway/Documents/Projects/HT/photo-library/run_server.py"],
      "env": {
        "PYTHONPATH": "/Users/casey-hemingway/Documents/Projects/HT/photo-library"
      }
    }
  }
}
```

**Important**: Use absolute paths, not relative paths!

## Step 4: Restart Claude Desktop

After updating the config, fully quit and restart Claude Desktop.

## Step 5: Test in Claude

Ask Claude:
```
Search my photos for people in rooms
```

```
Show me statistics about my photo archive
```

```
Find outdoor scenes
```

## Next Steps: Index Your Full Archive

Once the sample images work, update `config.yml` to point to your full photo archive:

```yaml
archive_path: "/path/to/your/actual/photo/archive"
```

Then run:
```bash
python3 test_indexing.py
```

## Troubleshooting

### "No module named 'mcp'" Error
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### "FAISS index not found" in Claude
The server needs to index images first. Run:
```bash
python3 test_indexing.py
```

### MCP Server Not Connecting
1. Check absolute paths in `claude_desktop_config.json`
2. Ensure `config.yml` exists
3. Check Claude Desktop logs: `~/Library/Logs/Claude/mcp*.log`
4. Check server logs: `./mcp-server.log`

### Slow on Intel Mac
Edit `config.yml`:
```yaml
clip:
  device: "cpu"  # Force CPU mode
  batch_size: 16  # Reduce batch size
```

## Performance Tips

### For Large Archives (10,000+ images)

1. **Use GPU acceleration**:
   - Apple Silicon: Automatically uses MPS
   - NVIDIA: Install `pip install faiss-gpu torch[cuda]`

2. **Incremental indexing**:
   After initial index, only new/modified images are processed:
   ```python
   # In test_indexing.py, change force=True to force=False
   stats = await indexer.index_archive(force=False)
   ```

3. **Increase batch size** (if you have RAM):
   ```yaml
   clip:
     batch_size: 64  # Process more images at once
   ```

## File Structure

```
photo-library/
├── src/                  # Source code
│   ├── server.py         # MCP server
│   ├── indexer.py        # Image indexing
│   ├── searcher.py       # Search functionality
│   ├── metadata.py       # EXIF extraction
│   ├── config.py         # Configuration
│   └── utils.py          # Utilities
├── data/                 # Generated (not in git)
│   ├── embeddings.faiss  # Vector index
│   ├── metadata.db       # SQLite database
│   └── thumbnails/       # Generated thumbnails
├── config.yml            # Your configuration
├── run_server.py         # MCP server entry point
├── test_indexing.py      # Test script
└── README.md             # Full documentation
```

## Common Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Test indexing
python3 test_indexing.py

# Run MCP server standalone (for debugging)
python3 run_server.py

# Deactivate virtual environment
deactivate
```

## Getting Help

- Check `README.md` for full documentation
- Review `mcp-server.log` for errors
- Open an issue on GitHub
