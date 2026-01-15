# Claude Desktop Setup Instructions

## Prerequisites

### Install uv (Python package manager)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Initialize the project
```bash
cd /path/to/photo-library
uv sync
```

## Setup Steps

### 1. Locate Claude Desktop Config File

The config file is at:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### 2. Update the Config

Open the file and add this configuration (update paths for your system):

```json
{
  "mcpServers": {
    "semantic-image-search": {
      "command": "/Users/YOUR_USERNAME/.local/bin/uv",
      "args": [
        "--directory",
        "/path/to/photo-library",
        "run",
        "python",
        "run_server.py"
      ],
      "env": {
        "KMP_DUPLICATE_LIB_OK": "TRUE"
      }
    }
  }
}
```

**Note:** Replace paths with your actual paths. Use `which uv` to find your uv path.

### 3. Restart Claude Desktop

1. Completely quit Claude Desktop (Cmd+Q)
2. Reopen Claude Desktop
3. The server should connect automatically

### 4. Verify Connection

In Claude, you should see the semantic-image-search server listed in the MCP section. You can test it by asking:

```
Show me statistics about my photo archive
```

or

```
Search my photos for outdoor scenes
```

## Troubleshooting

### Check Server Logs

If the server doesn't connect, check:
- Claude Desktop logs: `~/Library/Logs/Claude/mcp-server-semantic-image-search.log`
- Server logs: `/Users/casey-hemingway/Documents/Projects/HT/photo-library/mcp-server.log`

### Manual Test

Test the startup script manually:
```bash
cd /Users/casey-hemingway/Documents/Projects/HT/photo-library
./start_mcp_server.sh
```

Press Ctrl+C to stop. If it starts without errors, it should work with Claude Desktop.

### Common Issues

**"Server disconnected" error:**
- Make sure `config.yml` exists in the project directory
- Verify FAISS index was created (check for `data/embeddings.faiss`)
- Run `python3 test_indexing.py` to rebuild the index

**"No results found" in searches:**
- This is normal if the query doesn't match indexed images well
- Try broader queries like "outdoor" or "person"
- Check that images were actually indexed (look at archive stats)

## What's Available

Once connected, you can use these tools through Claude:

1. **search_images** - Semantic search with natural language
   - "Find images with people"
   - "Show me outdoor scenes from April 2024"
   - "Search for mountain landscapes"

2. **get_archive_stats** - View archive statistics
   - "What's in my photo archive?"
   - "Show me archive stats"

3. **get_image_info** - Get detailed metadata for specific images
   - "Get details for image abc123"

4. **reindex_archive** - Add new photos to the index
   - "Reindex my photo archive"
   - "Add new photos to the index"

## Next Steps

### Index Your Full Archive

Currently indexed: 51 images from the test directory

To index your full Himalayan Trust photo archive:

1. Edit `config.yml`
2. Change `archive_path` to your main photo directory
3. Run: `KMP_DUPLICATE_LIB_OK=TRUE python3 test_indexing.py`
4. Restart Claude Desktop

Enjoy searching your photos! 🏔️
