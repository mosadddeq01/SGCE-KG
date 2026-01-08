# KG Explorer App

This directory hosts a lightweight wrapper around `src/kg_gen/utils/template.html`. It provides:

- drag-and-drop rendering for existing `graph.json` files or saved template payloads;
- a minimal API service that invokes `KGGen` so you can generate graphs from raw text without altering the core library.

## Quick start

```bash
# 1. Install requirements in your environment
pip install -e .
pip install fastapi uvicorn[standard]

# 2. Launch the app server
uvicorn app.server:app --reload --port 8000
```

Then open http://localhost:8000/ in your browser. The upload tab works immediately. To generate new graphs, paste text or upload a `.txt` file and supply your personal OpenAI API key; all processing happens on this machine.

For deployment on a platform like Fly.io, Render, or Railway:

1. Ship the full repository (so the server can import `src.kg_gen`).
2. Install the same dependencies as above.
3. Start with `uvicorn app.server:app --host 0.0.0.0 --port $PORT` (most hosts expose `$PORT`).

Static hosting is also supported for the pure upload flow. Serve the `app/` directory together with `src/kg_gen/utils/template.html`, ensuring the template is reachable at `/template` relative to the site root.

## Notes

- The API never persists uploads or API keys. Everything runs in-memory per request.
- The download button provides the raw `graph.json` returned from `KGGen` so you can reuse it elsewhere.
- If you already have a pre-rendered visualization JSON (with `nodes`, `edges`, and `stats` fields), the app skips the server round trip and renders it directly in the browser.
