# kg-gen MCP Server: Agent Memory with Knowledge Graphs

This directory contains the Model Context Protocol (MCP) server for kg-gen, providing agents with persistent memory capabilities through knowledge graph extraction and storage.

## What is MCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open standard that enables AI applications to provide context to LLMs in a secure, controlled way. This kg-gen MCP server allows AI agents to:

- `add_memories`: **Extract and store memories** from unstructured text
- `retrieve_relevant_memories`: **Retrieve relevant memories** based on queries  

These tools are also provided for visibility:
- `visualize_memories`: **Visualize memory graphs** as interactive HTML
- `get_memory_stats`: **Get memory statistics** and manage storage

## Quick Start

### Installation

```bash
# Install kg-gen with MCP support
pip install kg-gen

# MCP dependencies are installed automatically when first needed
```

### Running the Server

```bash
# Start MCP server (clears memory by default for fresh start)
kggen mcp

# Keep existing memory
kggen mcp --keep-memory

# Use custom model and storage
kggen mcp --model gemini/gemini-2.0-flash --storage-path ./my_memory.json
```

### Configuration

The server can be configured via command line options or environment variables:

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `--model` | `KG_MODEL` | `openai/gpt-5` | Model to use for extraction |
| `--storage-path` | `KG_STORAGE_PATH` | `./kg_memory.json` | Path for memory storage |
| `--keep-memory` | `KG_CLEAR_MEMORY=false` | Clear memory | Keep existing memory on startup |
| N/A | `KG_API_KEY` or `OPENAI_API_KEY` | None | API key for model access |

## Available Tools

The MCP server provides four main tools for agent memory management:

### 1. `add_memories`
Extract and store memories from unstructured text.

```python
# Example usage in an MCP client
result = await client.call_tool("add_memories", {
    "text": "Alice works at Google as a software engineer. She graduated from MIT in 2020."
})
```

**Output:**
```
Successfully extracted and stored memories from text.
New memories: 4 entities, 3 relations
Total memories: 4 entities, 3 relations
Storage: Saved successfully
```

### 2. `retrieve_relevant_memories`
Query stored memories to find relevant information.

```python
result = await client.call_tool("retrieve_relevant_memories", {
    "query": "Alice work"
})
```

**Output:**
```
Relevant memories for 'Alice work':

Related entities (2):
- Alice
- Google

Related facts (2):
- Alice works at Google
- Alice is software engineer
```

### 3. `visualize_memories`
Generate an interactive HTML visualization of the memory graph.

```python
result = await client.call_tool("visualize_memories", {
    "output_filename": "my_memories.html"
})
```

**Output:**
```
Memory graph visualization saved to: /path/to/my_memories.html

Visualization contains 4 entities and 3 relations.
Open the HTML file in your browser to view the interactive graph.
```

### 4. `get_memory_stats`
Get statistics about stored memories.

```python
result = await client.call_tool("get_memory_stats", {})
```

**Output:**
```
Memory Statistics:
- Total Entities: 4
- Total Relations: 3
- Edge Types: 2
- Storage Path: ./kg_memory.json
- Entity Clusters: 0
- Edge Clusters: 0
```

## Memory Management

### Fresh Start vs. Persistent Memory

By default, `kggen mcp` clears existing memory on startup to provide a clean slate for each session:

```bash
# Fresh start (default) - clears existing memory
kggen mcp

# Persistent memory - keeps existing memory
kggen mcp --keep-memory
```

### Storage Path Resolution

Storage paths are resolved relative to where you call the command:

```bash
# From /home/user/project/
kggen mcp                                    # → /home/user/project/kg_memory.json
kggen mcp --storage-path ./data/memory.json  # → /home/user/project/data/memory.json
kggen mcp --storage-path /abs/path/mem.json  # → /abs/path/mem.json
```

## Integration Examples

### Claude Desktop

Add to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "kg-memory": {
      "command": "kggen",
      "args": ["mcp", "--keep-memory"],
      "env": {
        "KG_MODEL": "openai/gpt-4o",
        "OPENAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### Custom MCP Client

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_kg_memory():
    server_params = StdioServerParameters(
        command="kggen",
        args=["mcp"],
        env={"OPENAI_API_KEY": "your-api-key"}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # Add some memories
            result = await session.call_tool("add_memories", {
                "text": "The meeting is scheduled for tomorrow at 3 PM in the conference room."
            })
            print(result.content[0].text)
            
            # Retrieve memories
            result = await session.call_tool("retrieve_relevant_memories", {
                "query": "meeting time"
            })
            print(result.content[0].text)

# Run the example
asyncio.run(use_kg_memory())
```

## Advanced Usage

### Direct Server Execution

You can also run the server directly with `fastmcp`:

```bash
# Basic usage
fastmcp run server.py

# With environment variables
KG_MODEL=gemini/gemini-2.0-flash KG_CLEAR_MEMORY=true fastmcp run server.py
```

### Custom Models

The server supports any model compatible with [LiteLLM](https://docs.litellm.ai/docs/providers):

```bash
# OpenAI models
kggen mcp --model openai/gpt-4o-mini

# Anthropic models  
kggen mcp --model anthropic/claude-3-sonnet-20240229

# Google models
kggen mcp --model gemini/gemini-2.0-flash

# Local models via Ollama
kggen mcp --model ollama_chat/llama3.2:3b
```

### Memory Persistence Patterns

```bash
# Development: Fresh memory each session
kggen mcp

# Production: Persistent memory across sessions
kggen mcp --keep-memory --storage-path /persistent/path/memory.json

# Project-specific: Memory per project
cd /path/to/project
kggen mcp --keep-memory  # Uses ./kg_memory.json in project directory
```

## Testing

Run the comprehensive test suite:

```bash
# Run all MCP tests
pytest mcp/tests/ -v

# Run specific test categories
pytest mcp/tests/test_mcp.py -v          # Integration tests
pytest mcp/tests/test_unit.py -v         # Unit tests  
pytest mcp/tests/test_memory_file_creation.py -v  # Memory management tests
```

## Architecture

The MCP server consists of:

- **`server.py`**: Main MCP server implementation with FastMCP
- **Memory Management**: Automatic loading/saving of knowledge graphs
- **Path Resolution**: Proper handling of relative vs absolute storage paths
- **Tool Interface**: Four core tools for memory operations
- **Error Handling**: Graceful handling of API failures and file system issues
