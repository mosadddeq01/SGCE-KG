# MCP Server Tests

This directory contains tests for the kg-gen MCP (Model Context Protocol) server.

## Test Structure

- `test_mcp.py` - Integration tests for MCP tools using real MCP client-server communication
- `test_unit.py` - Unit tests for individual server functions
- `run_all_tests.py` - Test runner script

## Running Tests

### Prerequisites

1. Install test dependencies:
```bash
# Install MCP dependencies
uv sync --group mcp

# Install dev dependencies (includes pytest)
uv sync --group dev
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your-api-key-here"
# Optional: Use a cheaper model for testing
export KG_MODEL="openai/gpt-4o-mini"
```

### Running All Tests

```bash
# From the mcp/tests directory
python run_all_tests.py

# Or using pytest directly from the project root
pytest mcp/tests/ -v
```

### Running Specific Test Files

```bash
# Run only MCP integration tests
pytest mcp/tests/test_mcp.py -v

# Run only unit tests
pytest mcp/tests/test_unit.py -v
```

### Running Specific Tests

```bash
# Run a specific test function
pytest mcp/tests/test_mcp.py::test_add_memories_tool -v

# Run tests matching a pattern
pytest mcp/tests/ -k "memory" -v
```

## Test Coverage

### MCP Integration Tests (`test_mcp.py`)

Tests the full MCP client-server communication for all tools:

- `add_memories` - Adding memories from text
- `retrieve_relevant_memories` - Querying stored memories  
- `get_memory_stats` - Getting memory statistics
- `visualize_memories` - Generating HTML visualizations
- Memory persistence across server restarts
- Memory aggregation from multiple additions
- Error handling for edge cases

### Unit Tests (`test_unit.py`)

Tests individual server functions in isolation:

- `initialize_kg_gen()` - Server initialization
- `load_memory_graph()` - Loading graphs from storage
- `save_memory_graph()` - Saving graphs to storage
- Environment variable handling
- Error handling for corrupted files

## Test Environment

Tests use temporary storage files to avoid interfering with real data. Each test gets its own isolated storage location.

## Troubleshooting

### Common Issues

1. **Missing API Key**: Tests require an OpenAI API key. Set `OPENAI_API_KEY` environment variable.

2. **MCP Dependencies**: Make sure `fastmcp` and `mcp` packages are installed:
   ```bash
   uv sync --group mcp
   ```

3. **Async Test Issues**: Tests use pytest-asyncio for async test support. Make sure it's installed:
   ```bash
   uv sync --group dev
   ```

### Test Failures

If tests fail, check:

1. API key is valid and has credits
2. Network connectivity for LLM calls
3. File permissions for temporary storage
4. MCP server can start properly

### Skipping Expensive Tests

To skip tests that make actual LLM calls (for faster development):

```bash
pytest mcp/tests/test_unit.py -v
```

Unit tests use mocks and don't make real API calls.
