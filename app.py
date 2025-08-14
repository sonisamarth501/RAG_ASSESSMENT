import streamlit as st
import os
import git
import shutil
from pathlib import Path
import tempfile
from typing import List, Dict, Any
import re
from dataclasses import dataclass

# Core libraries
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import ollama

# Code parsing libraries
import ast
import tokenize
import io
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer_for_filename
from pygments.formatters import TerminalFormatter
from pygments.util import ClassNotFound

# Language detection - using file extensions and content patterns

@dataclass
class CodeSnippet:
    """Represents a code snippet with metadata"""
    content: str
    language: str
    file_path: str
    start_line: int
    end_line: int
    function_name: str = None
    class_name: str = None
    docstring: str = None
    imports: List[str] = None

class CodeParser:
    """Handles parsing of different programming languages"""
    
    SUPPORTED_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.sql': 'sql',
        '.sh': 'bash',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css',
        '.md': 'markdown',
        '.rst': 'rst'
    }
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def detect_language(self, file_path: str, content: str) -> str:
        """Detect programming language from file extension or content"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in self.SUPPORTED_EXTENSIONS:
            return self.SUPPORTED_EXTENSIONS[file_ext]
        
        # Fallback: try to guess from content patterns
        content_lower = content.lower()
        
        # Simple content-based detection
        if 'def ' in content and 'import ' in content:
            return 'python'
        elif 'function' in content and ('var ' in content or 'const ' in content):
            return 'javascript'
        elif 'public class' in content and 'import java' in content:
            return 'java'
        elif '#include' in content and ('int main' in content or 'void' in content):
            return 'cpp'
        elif 'func ' in content and 'package ' in content:
            return 'go'
        elif '<!DOCTYPE html>' in content or '<html' in content:
            return 'html'
        elif '{' in content and '"' in content and ':' in content:
            return 'json'
        else:
            return 'text'
    
    def extract_python_functions(self, content: str, file_path: str) -> List[CodeSnippet]:
        """Extract functions and classes from Python code"""
        snippets = []
        
        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
                    
                    snippet_content = '\n'.join(lines[start_line:end_line])
                    docstring = ast.get_docstring(node) if hasattr(ast, 'get_docstring') else None
                    
                    snippet = CodeSnippet(
                        content=snippet_content,
                        language='python',
                        file_path=file_path,
                        start_line=start_line + 1,
                        end_line=end_line,
                        function_name=node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None,
                        class_name=node.name if isinstance(node, ast.ClassDef) else None,
                        docstring=docstring
                    )
                    snippets.append(snippet)
                    
        except SyntaxError as e:
            st.warning(f"Syntax error in {file_path}: {e}")
        
        return snippets
    
    def extract_generic_functions(self, content: str, file_path: str, language: str) -> List[CodeSnippet]:
        """Extract functions using regex patterns for other languages"""
        snippets = []
        lines = content.split('\n')
        
        # Define function patterns for different languages
        patterns = {
            'javascript': [
                r'function\s+(\w+)\s*\(',
                r'const\s+(\w+)\s*=\s*\(',
                r'(\w+)\s*:\s*function\s*\(',
                r'(\w+)\s*=>\s*{?'
            ],
            'java': [
                r'(public|private|protected)?\s*(static)?\s*\w+\s+(\w+)\s*\('
            ],
            'cpp': [
                r'\w+\s+(\w+)\s*\([^)]*\)\s*{'
            ],
            'go': [
                r'func\s+(\w+)\s*\('
            ]
        }
        
        if language in patterns:
            for pattern in patterns[language]:
                for i, line in enumerate(lines):
                    match = re.search(pattern, line)
                    if match:
                        # Extract function block (simple heuristic)
                        start_line = i
                        end_line = min(i + 20, len(lines))  # Approximate end
                        
                        snippet_content = '\n'.join(lines[start_line:end_line])
                        
                        snippet = CodeSnippet(
                            content=snippet_content,
                            language=language,
                            file_path=file_path,
                            start_line=start_line + 1,
                            end_line=end_line,
                            function_name=match.group(-1)  # Last captured group
                        )
                        snippets.append(snippet)
        
        return snippets

class GitHubProcessor:
    """Handles GitHub repository processing"""
    
    def __init__(self):
        self.temp_dir = None
        self.code_parser = CodeParser()
        
    def clone_repository(self, repo_url: str) -> str:
        """Clone GitHub repository to temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            git.Repo.clone_from(repo_url, self.temp_dir)
            st.success(f"Repository cloned successfully to {self.temp_dir}")
            return self.temp_dir
        except Exception as e:
            st.error(f"Error cloning repository: {e}")
            return None
    
    def scan_repository(self, repo_path: str) -> List[Dict[str, Any]]:
        """Scan repository for code files and documentation"""
        documents = []
        
        for root, dirs, files in os.walk(repo_path):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env']]
            
            for file in files:
                if file.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    language = self.code_parser.detect_language(file_path, content)
                    
                    # Process different file types
                    if language in ['python']:
                        snippets = self.code_parser.extract_python_functions(content, relative_path)
                        for snippet in snippets:
                            documents.append({
                                'content': snippet.content,
                                'metadata': {
                                    'file_path': relative_path,
                                    'language': snippet.language,
                                    'type': 'function' if snippet.function_name else 'class',
                                    'name': snippet.function_name or snippet.class_name,
                                    'docstring': snippet.docstring,
                                    'start_line': snippet.start_line,
                                    'end_line': snippet.end_line
                                }
                            })
                    elif language in ['javascript', 'java', 'cpp', 'go']:
                        snippets = self.code_parser.extract_generic_functions(content, relative_path, language)
                        for snippet in snippets:
                            documents.append({
                                'content': snippet.content,
                                'metadata': {
                                    'file_path': relative_path,
                                    'language': snippet.language,
                                    'type': 'function',
                                    'name': snippet.function_name,
                                    'start_line': snippet.start_line,
                                    'end_line': snippet.end_line
                                }
                            })
                    elif language in ['markdown', 'rst', 'text']:
                        # Process documentation files
                        chunks = self._chunk_text(content, 1000, 200)
                        for i, chunk in enumerate(chunks):
                            documents.append({
                                'content': chunk,
                                'metadata': {
                                    'file_path': relative_path,
                                    'language': language,
                                    'type': 'documentation',
                                    'chunk_id': i
                                }
                            })
                    else:
                        # Process as whole file for other types
                        if len(content) < 5000:  # Only small files
                            documents.append({
                                'content': content,
                                'metadata': {
                                    'file_path': relative_path,
                                    'language': language,
                                    'type': 'file'
                                }
                            })
                            
                except Exception as e:
                    st.warning(f"Error processing {relative_path}: {e}")
                    
        return documents
    
    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into chunks with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap if end < len(text) else end
            
        return chunks
    
    def cleanup(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

class VectorStore:
    """Handles vector storage and retrieval using ChromaDB"""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.collection = None
    
    def create_collection(self, collection_name: str):
        """Create or get collection"""
        try:
            self.collection = self.client.delete_collection(name=collection_name)
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to vector store"""
        contents = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Add in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            self.collection.add(
                documents=batch_contents,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
    
    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for relevant documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

class RAGSystem:
    """Main RAG system combining retrieval and generation"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.github_processor = GitHubProcessor()
        
    def process_repository(self, repo_url: str, collection_name: str):
        """Process GitHub repository and create vector store"""
        # Clone repository
        repo_path = self.github_processor.clone_repository(repo_url)
        if not repo_path:
            return False
            
        try:
            # Scan and process files
            st.info("Scanning repository for code and documentation...")
            documents = self.github_processor.scan_repository(repo_path)
            
            if not documents:
                st.error("No processable documents found in repository")
                return False
            
            st.info(f"Found {len(documents)} code snippets and documents")
            
            # Create vector store
            st.info("Creating vector embeddings...")
            self.vector_store.create_collection(collection_name)
            self.vector_store.add_documents(documents)
            
            st.success(f"Successfully processed repository with {len(documents)} documents")
            return True
            
        finally:
            # Cleanup
            self.github_processor.cleanup()
    
    def query(self, question: str, n_results: int = 5) -> str:
        """Query the RAG system"""
        # Retrieve relevant documents
        search_results = self.vector_store.search(question, n_results)
        
        if not search_results['documents'][0]:
            return "No relevant information found."
        
        # Prepare context
        context_docs = []
        for i, (doc, metadata) in enumerate(zip(search_results['documents'][0], search_results['metadatas'][0])):
            context_docs.append(f"Document {i+1} ({metadata.get('file_path', 'unknown')}):\n{doc}")
        
        context = "\n\n".join(context_docs)
        
        # Generate response using Ollama
        prompt = f"""You are a code documentation expert. Based on the following context from a GitHub repository, answer the user's question about the codebase.

Context:
{context}

Question: {question}

Please provide a detailed answer that includes:
1. Direct answer to the question
2. Relevant code examples if applicable
3. File paths and line numbers where relevant
4. Implementation details and explanations

Answer:"""

        try:
            response = ollama.generate(
                model='llama3.2',
                prompt=prompt,
                options={'temperature': 0.1}  # Lower temperature for more factual responses
            )
            return response['response']
        except Exception as e:
            return f"Error generating response: {e}"

# Streamlit App
def main():
    st.set_page_config(page_title="Code Documentation RAG", layout="wide")
    
    st.title("🚀 Code Documentation RAG System")
    st.markdown("**Intelligent code repository analysis and Q&A system**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check Ollama connection
        try:
            ollama.list()
            st.success("✅ Ollama connected")
        except:
            st.error("❌ Ollama not connected. Please ensure Ollama is running.")
            st.stop()
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Enter a GitHub repository URL
        2. Click 'Process Repository'
        3. Wait for processing to complete
        4. Ask questions about the code
        """)
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    if 'repository_processed' not in st.session_state:
        st.session_state.repository_processed = False
    if 'current_repo' not in st.session_state:
        st.session_state.current_repo = ""
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Repository Input")
        repo_url = st.text_input(
            "GitHub Repository URL",
            placeholder="https://github.com/username/repository",
            help="Enter the full GitHub repository URL"
        )
        
        if st.button("🔄 Process Repository", type="primary"):
            if repo_url:
                if repo_url != st.session_state.current_repo:
                    st.session_state.repository_processed = False
                    st.session_state.current_repo = repo_url
                
                with st.spinner("Processing repository..."):
                    collection_name = repo_url.split('/')[-1].replace('.git', '')
                    success = st.session_state.rag_system.process_repository(repo_url, collection_name)
                    st.session_state.repository_processed = success
            else:
                st.error("Please enter a repository URL")
    
    with col2:
        st.header("📊 System Status")
        if st.session_state.repository_processed:
            st.success("✅ Repository processed")
            st.info(f"📂 Current repo: {st.session_state.current_repo.split('/')[-1]}")
        else:
            st.warning("⏳ No repository processed")
    
    # Query interface
    if st.session_state.repository_processed:
        st.header("❓ Ask Questions")
        
        # Sample questions
        with st.expander("💡 Sample Questions"):
            sample_questions = [
                "What are the main functions in this repository?",
                "How does the authentication system work?",
                "Show me examples of API endpoints",
                "What are the main classes and their purposes?",
                "How is error handling implemented?",
                "What are the dependencies of this project?",
                "Explain the database schema",
                "How to run tests in this project?"
            ]
            
            for q in sample_questions:
                if st.button(q, key=f"sample_{q}"):
                    st.session_state.current_question = q
        
        # Query input
        question = st.text_area(
            "Your Question",
            value=st.session_state.get('current_question', ''),
            height=100,
            placeholder="Ask anything about the codebase..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            n_results = st.slider("Results to retrieve", 1, 10, 5)
        
        if st.button("🔍 Get Answer", type="primary"):
            if question:
                with st.spinner("Searching codebase and generating answer..."):
                    answer = st.session_state.rag_system.query(question, n_results)
                    
                    st.header("💡 Answer")
                    st.markdown(answer)
            else:
                st.error("Please enter a question")

if __name__ == "__main__":
    main()