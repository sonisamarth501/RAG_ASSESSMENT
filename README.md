# 🚀 Code Documentation RAG System

An intelligent Retrieval Augmented Generation (RAG) system that analyzes GitHub repositories and provides natural language Q&A about codebases using vector embeddings and large language models.

## 🌟 Features

### Core Functionality
- **GitHub Repository Analysis**: Clone and analyze any public GitHub repository
- **Multi-language Support**: Python, JavaScript/TypeScript, Java, C/C++, Go, Rust, and more
- **Intelligent Code Parsing**: Extract functions, classes, and documentation with metadata
- **Vector Search**: Semantic search using sentence transformers
- **Natural Language Q&A**: Ask questions about code in plain English
- **Real-time Processing**: Progress tracking and status updates

### Advanced Features
- **Code Complexity Analysis**: Calculate complexity scores for functions and classes
- **Documentation Processing**: Parse README files, comments, and docstrings
- **Smart Chunking**: Intelligent text segmentation respecting code structure
- **Binary File Detection**: Skip non-text files automatically
- **Encoding Handling**: Support for multiple text encodings
- **Error Recovery**: Graceful handling of parsing errors

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GitHub Repo   │───▶│   Code Parser    │───▶│  Vector Store   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │◀───│   RAG System     │◀───│   ChromaDB      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Ollama LLM     │
                       └──────────────────┘
```

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- At least 4GB RAM (8GB recommended)
- 2GB disk space for dependencies and models

### Required Software
1. **Ollama**: For running local language models
   ```bash
   # Install Ollama (https://ollama.ai)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the required model
   ollama pull llama3.2
   ```

2. **Git**: For repository cloning
   ```bash
   # Ubuntu/Debian
   sudo apt-get install git
   
   # macOS
   brew install git
   
   # Windows: Download from https://git-scm.com/
   ```

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/code-documentation-rag
cd code-documentation-rag
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
# Check if Ollama is running
ollama list

# Test Streamlit
streamlit hello
```

## 🎯 Quick Start

### 1. Start the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 2. Process a Repository
1. Enter a GitHub repository URL (e.g., `https://github.com/username/repo`)
2. Click "Process Repository"
3. Wait for processing to complete (progress bar will show status)

### 3. Ask Questions
Once processing is complete, you can ask questions like:
- "What are the main functions in this repository?"
- "How does the authentication system work?"
- "Show me examples of API endpoints"
- "What are the main classes and their purposes?"

## 📖 Usage Examples

### Repository Processing
```python
# The system supports various URL formats:
https://github.com/username/repository
github.com/username/repository
username/repository
```

### Sample Questions

#### General Code Analysis
- "What is the overall architecture of this project?"
- "What are the main entry points?"
- "How is error handling implemented?"

#### Function-Specific Queries
- "Show me all functions that handle user authentication"
- "What functions interact with the database?"
- "Find functions with high complexity scores"

#### Documentation Queries
- "How do I set up this project locally?"
- "What are the API endpoints available?"
- "What dependencies does this project have?"

#### Language-Specific Questions
- "Show me all Python classes in this repository"
- "What JavaScript functions handle API calls?"
- "How are Go routines used in this codebase?"

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the project root:
```env
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2

# ChromaDB Configuration
CHROMA_DB_PATH=./chroma_db

# Processing Configuration
MAX_FILE_SIZE=1048576  # 1MB in bytes
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
```

### Model Configuration
You can use different Ollama models:
```bash
# Available models (choose based on your hardware)
ollama pull llama3.2      # Recommended, good balance
ollama pull codellama     # Optimized for code
ollama pull llama2        # Smaller, faster
ollama pull mistral       # Alternative option
```

Update the model name in the code:
```python
response = ollama.generate(
    model='your-preferred-model',  # Change this
    prompt=prompt,
    options={'temperature': 0.1}
)
```

## 🛠️ Supported Languages

### Fully Supported (with function/class extraction)
- **Python** (.py) - Functions, classes, docstrings, imports
- **JavaScript/TypeScript** (.js, .jsx, .ts, .tsx) - Functions, classes, methods
- **Java** (.java) - Methods, classes, interfaces
- **C/C++** (.c, .cpp, .h, .hpp) - Functions, classes, structs
- **Go** (.go) - Functions, structs, interfaces
- **Rust** (.rs) - Functions, structs, enums, traits

### Partially Supported (content-based analysis)
- **Ruby** (.rb), **PHP** (.php), **Swift** (.swift)
- **Kotlin** (.kt), **Scala** (.scala), **R** (.r)
- **Shell Scripts** (.sh), **SQL** (.sql)

### Documentation & Configuration
- **Markdown** (.md), **reStructuredText** (.rst)
- **YAML** (.yml, .yaml), **JSON** (.json), **XML** (.xml)
- **HTML/CSS** (.html, .css, .scss), **Dockerfile**

## 📊 Performance Optimization

### For Large Repositories
1. **Selective Processing**: The system automatically skips:
   - Binary files (.png, .jpg, .exe, etc.)
   - Build directories (node_modules, target, dist)
   - Cache directories (__pycache__, .cache)
   - Version control files (.git)

2. **Memory Management**: 
   - Files are processed in batches
   - Large files (>1MB) are automatically skipped
   - Vector embeddings are created in chunks

3. **Processing Tips**:
   - Start with smaller repositories to test
   - Monitor system memory usage
   - Use SSD storage for better ChromaDB performance

## 🔧 Troubleshooting

### Common Issues

#### 1. Ollama Connection Error
```
Error: Ollama not connected
```
**Solution**:
```bash
# Check if Ollama is running
ollama serve

# Verify model is installed
ollama list

# Test model
ollama run llama3.2
```

#### 2. ChromaDB Permission Error
```
Error: Permission denied accessing ./chroma_db
```
**Solution**:
```bash
# Fix permissions
chmod -R 755 ./chroma_db

# Or use different directory
mkdir ~/chroma_data
# Update CHROMA_DB_PATH in .env
```

#### 3. Git Clone Failed
```
Error cloning repository: [authentication required]
```
**Solution**:
- Ensure repository is public
- Check internet connection
- Verify repository URL format

#### 4. Memory Issues
```
Error: Out of memory during processing
```
**Solution**:
- Reduce batch size in configuration
- Process smaller repositories first
- Close other memory-intensive applications
- Increase system RAM if possible

#### 5. Model Loading Error
```
Error: Model not found
```
**Solution**:
```bash
# Pull the required model
ollama pull llama3.2

# Check available models
ollama list

# Restart Ollama service
ollama serve
```

## 🧪 Development

### Project Structure
```
code-documentation-rag/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .env                  # Environment variables (create this)
├── chroma_db/            # ChromaDB storage (auto-created)
├── tests/                # Unit tests (optional)
└── docs/                 # Additional documentation
```

### Key Classes
- **`CodeSnippet`**: Data structure for code snippets with metadata
- **`EnhancedCodeParser`**: Language-aware code parsing and analysis
- **`EnhancedGitHubProcessor`**: Repository cloning and file processing
- **`EnhancedVectorStore`**: ChromaDB interface for vector operations
- **`RAGSystem`**: Main orchestrator combining all components

### Adding New Languages
1. Add file extensions to `SUPPORTED_EXTENSIONS`
2. Add detection patterns to `detect_language()`
3. Add extraction patterns to `extract_generic_functions()`
4. Test with sample repositories

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## 🤝 Contributing

I welcome contributions!

### Areas for Improvement
- [ ] Support for more programming languages
- [ ] Better code complexity analysis
- [ ] Integration with more LLM providers
- [ ] Caching and incremental processing
- [ ] API endpoint for programmatic access
- [ ] Docker containerization
- [ ] Better error handling and logging
- [ ] Performance optimizations


## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the web interface
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Ollama](https://ollama.ai/) for local LLM inference
- [GitPython](https://gitpython.readthedocs.io/) for repository handling

## 📬 Support

- **Issues**: [Linedin](https://www.linkedin.com/in/sonisamarth501/)
- **Email**: sonisamarth501@gmail.com

## 🗺️ Roadmap

### Version 2.0 (Planned)
- [ ] Multi-repository analysis
- [ ] Code similarity detection
- [ ] Integration with IDEs (VS Code extension)
- [ ] Real-time repository monitoring
- [ ] Advanced analytics dashboard
- [ ] Team collaboration features

### Version 2.1 (Future)
- [ ] AI-powered code suggestions
- [ ] Automated documentation generation
- [ ] Security vulnerability detection
- [ ] Code quality metrics
- [ ] Integration with CI/CD pipelines

---

**Happy coding! 🎉**