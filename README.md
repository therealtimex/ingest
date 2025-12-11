# ingest

High-quality document processing CLI for RAG pipelines. Process PDFs, Office documents, images, and more into markdown, JSON, HTML, or RAG-optimized chunks.

## Features

✅ **Standalone CLI Command** - Simple `ingest` command  
✅ **4 Output Formats** - markdown, json, html, chunks (RAG-optimized)  
✅ **3 Converters** - pdf, table, ocr specialized processing  
✅ **LLM Enhancement** - Optional AI boost (81% → 91% table accuracy)  
✅ **Multi-Worker** - Parallel batch processing  
✅ **20+ Options** - Full control over processing  

## Quick Start

### Installation

#### Using uv (recommended - fastest!)

```bash
# Basic installation (fast, lightweight)
uv pip install ingest-cli

# With marker-pdf for high-quality processing
uv pip install ingest-cli[marker]

# With LLM support
uv pip install ingest-cli[llm]

# Full installation (everything)
uv pip install ingest-cli[full]
```

#### Using pip

```bash
# Basic installation
pip install ingest-cli

# With marker-pdf
pip install ingest-cli[marker]

# Full installation
pip install ingest-cli[full]
```

#### Install from source

```bash
git clone https://github.com/therealtimex/ingest.git
cd ingest

# Basic installation (lightweight, no marker-pdf)
uv pip install -e .

# With marker-pdf for high-quality processing
uv pip install -e ".[marker]"

# Full installation with all features
uv pip install -e ".[full]"
```

### Basic Usage

```bash
# Process a document
ingest document.pdf

# Process for RAG
ingest ./documents --output-format chunks --batch-mode

# Extract tables with LLM
ingest report.pdf --converter-type table --use-llm

# View help
ingest --help
```

## Common Use Cases

### 1. RAG System Preparation

```bash
ingest ./knowledge_base \
    --output-format chunks \
    --batch-mode \
    --workers 4
```

**Output**: Pre-chunked JSON optimized for embeddings and retrieval.

### 2. Table Extraction

```bash
ingest financial_reports/ \
    --converter-type table \
    --use-llm \
    --output-format json \
    --batch-mode
```

**Output**: High-accuracy table data in JSON format.

### 3. OCR Scanned Documents

```bash
ingest scanned_docs/ \
    --force-ocr \
    --output-format markdown \
    --batch-mode
```

**Output**: Clean markdown from scanned PDFs.

## Output Formats

- **markdown**: Clean markdown with proper formatting
- **json**: Structured JSON with full metadata
- **html**: Web-ready HTML with embedded images
- **chunks**: RAG-optimized pre-chunked JSON for vector databases

## Performance

| Workers | VRAM | Throughput (H100) |
|---------|------|-------------------|
| 1 | 5GB | ~30 pages/sec |
| 4 | 20GB | ~120 pages/sec |
| 8 | 40GB | ~240 pages/sec |

## Requirements

- Python 3.10+
- Optional: GPU for faster processing (CPU mode available)

## Environment Variables

```bash
# PyTorch device
export TORCH_DEVICE=cuda  # or cpu, mps

# LLM API keys (optional, for enhanced accuracy)
export GOOGLE_API_KEY="your-gemini-key"
export ANTHROPIC_API_KEY="your-claude-key"
export OPENAI_API_KEY="your-openai-key"
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

- Issues: [GitHub Issues](https://github.com/therealtimex/ingest/issues)
- Repository: [github.com/therealtimex/ingest](https://github.com/therealtimex/ingest)

---

Built with ❤️ by [RealTimeX](https://realtimex.ai)
