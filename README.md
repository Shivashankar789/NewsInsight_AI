# News Research Tool 📈

A RAG (Retrieval-Augmented Generation) based AI-powered tool that enables intelligent question-answering from news articles. Built with LangChain, OpenAI, FAISS, and Streamlit.

## 🌟 Features

- **URL-based Content Loading**: Load and process content from up to 3 news article URLs simultaneously
- **Intelligent Text Processing**: Automatic text splitting and chunking for optimal processing
- **Vector Search**: FAISS-based vector store for efficient similarity search
- **RAG Implementation**: Retrieval-Augmented Generation for accurate, source-cited answers
- **Interactive UI**: Clean Streamlit interface for easy interaction
- **Source Attribution**: Get answers with references to original sources

## 🏗️ Architecture

The application implements a complete RAG pipeline:

1. **Document Loading**: Uses `UnstructuredURLLoader` to fetch content from news URLs
2. **Text Splitting**: `RecursiveCharacterTextSplitter` chunks text intelligently with configurable separators
3. **Embeddings**: OpenAI embeddings convert text chunks into vector representations
4. **Vector Store**: FAISS index stores and retrieves similar content efficiently
5. **Question Answering**: `RetrievalQAWithSourcesChain` generates answers with source citations

## 📋 Prerequisites

- Python 3.7+
- OpenAI API Key
- Internet connection for loading news articles

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd 2_news_research_tool_project
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## 💻 Usage

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. The application will open in your default web browser

3. **Process URLs**:
   - Enter up to 3 news article URLs in the sidebar
   - Click "Process URLs" to load and index the content
   - Wait for the processing steps to complete:
     - Data Loading ✅
     - Text Splitting ✅
     - Embedding Vector Building ✅

4. **Ask Questions**:
   - Once processing is complete, enter your question in the text input
   - The system will retrieve relevant information and generate an answer
   - View the answer along with source references

## 📦 Dependencies

- **langchain** (0.0.284): Framework for LLM applications
- **streamlit** (1.22.0): Web application framework
- **openai** (0.28.0): OpenAI API client
- **faiss-cpu** (1.7.4): Facebook AI Similarity Search
- **unstructured** (0.9.2): Document parsing and loading
- **tiktoken** (0.4.0): Token counting for OpenAI models
- **python-dotenv** (1.0.0): Environment variable management
- **python-magic** libraries: File type detection

## 📁 Project Structure

```
2_news_research_tool_project/
├── main.py                          # Main Streamlit application
├── requirements.txt                 # Project dependencies
├── .env                            # Environment variables (create this)
├── faiss_store_openai.pkl          # Saved FAISS vector store (generated)
└── notebooks/                      # Jupyter notebooks for experimentation
    ├── faiss_tutorial.ipynb        # FAISS vector search tutorial
    ├── retrieval.ipynb             # RAG retrieval implementation
    ├── text_loaders_splitters.ipynb # Document loaders and text splitting
    ├── movies.csv                  # Sample data for testing
    ├── nvda_news_1.txt            # Sample news article
    ├── sample_text.csv            # Sample text data
    ├── chunk_size.jpg             # Documentation image
    └── vector_index.pkl           # Sample vector index
```

## 🔧 Configuration

### Text Splitter Settings
The application uses `RecursiveCharacterTextSplitter` with:
- **Chunk Size**: 1000 characters
- **Separators**: `['\n\n', '\n', '.', ',']`
- Optimized for maintaining context while creating manageable chunks

### LLM Settings
- **Model**: OpenAI (via LangChain)
- **Temperature**: 0.9 (for creative responses)
- **Max Tokens**: 500

## 🧪 Notebooks

The `notebooks/` directory contains educational Jupyter notebooks:

1. **faiss_tutorial.ipynb**: Learn FAISS vector indexing with sentence transformers
2. **retrieval.ipynb**: Understand RAG implementation with LangChain
3. **text_loaders_splitters.ipynb**: Explore different document loaders (TextLoader, CSVLoader, URLLoader)

## 🎯 Use Cases

- **Research Assistance**: Quickly extract information from multiple news sources
- **Fact Checking**: Cross-reference information across articles
- **News Analysis**: Ask analytical questions about current events
- **Content Summarization**: Get concise answers from lengthy articles

## ⚙️ How It Works

1. **URL Processing Phase**:
   - Users input news article URLs
   - Content is fetched and loaded into document format
   - Text is split into optimal chunks
   - OpenAI embeddings are created for each chunk
   - FAISS index is built and saved to disk

2. **Question Answering Phase**:
   - User submits a question
   - Question is embedded using the same embedding model
   - FAISS performs similarity search to find relevant chunks
   - LLM generates answer based on retrieved context
   - Answer is displayed with source URLs

## 🔐 Security Notes

- Never commit your `.env` file or expose your OpenAI API key
- The FAISS index (`faiss_store_openai.pkl`) is saved locally and contains processed article data
- Consider rate limits when processing multiple URLs

## 🐛 Troubleshooting

- **Installation Issues**: Ensure you have the correct version of Python and all system dependencies
- **Magic Library Errors** (Windows): The requirements include `python-magic-bin` for Windows compatibility
- **API Errors**: Verify your OpenAI API key is correctly set in the `.env` file
- **Loading Errors**: Some URLs may require additional parsing - ensure articles are publicly accessible

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Support for more URL sources
- Enhanced text splitting strategies
- Additional embedding models
- Better error handling
- UI/UX improvements

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [OpenAI](https://openai.com/)
- Vector search by [FAISS](https://github.com/facebookresearch/faiss)
- UI by [Streamlit](https://streamlit.io/)

---

**Note**: This is an educational project demonstrating RAG implementation for news article analysis.
