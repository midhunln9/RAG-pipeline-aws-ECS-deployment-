# Financial Compliance RAG Chatbot UI

A professional Streamlit-based chatbot interface for querying a Financial Compliance RAG (Retrieval-Augmented Generation) system.

## Features

- 🎯 **Multi-turn Conversations** - Maintain context across multiple messages using session IDs
- 📚 **Source Citations** - View retrieved source documents with metadata and expandable content
- 🎨 **Professional UI** - Clean, modern interface optimized for interviews and presentations
- ⚡ **Real-time Responses** - Stream responses from the RAG backend API
- 🛡️ **Error Handling** - Graceful handling of API connection issues

## Setup

### Prerequisites

- Python 3.9+
- RAG API server running on localhost:8000 (or configured URL)

### Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the API URL (optional):
```bash
cp .env.example .env
# Edit .env if your API is not at http://localhost:8000
```

## Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Usage

1. **Start a Conversation**: Type your question in the input field at the bottom
2. **Send Message**: Click "Send" or press Enter
3. **View Sources**: Scroll down to see cited source documents below each response
4. **Expand Details**: Click on source items to expand and view full content and metadata
5. **New Conversation**: Click "🔄 New Conversation" to start a fresh session

## Project Structure

```
UI/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variable template
└── README.md             # This file
```

## API Integration

The UI communicates with the RAG API using the `/ask` endpoint:

**Request:**
```json
{
  "query": "What is financial compliance?",
  "session_id": "uuid-string"
}
```

**Response:**
```json
{
  "response": "Financial compliance refers to...",
  "session_id": "uuid-string",
  "sources": [
    {
      "content": "Document text...",
      "metadata": {
        "source": "compliance_guide.pdf",
        "page": 5
      }
    }
  ]
}
```

## Environment Variables

- `API_BASE_URL` - Base URL of the RAG API (default: `http://localhost:8000`)

## Customization

### Styling

The app uses custom CSS injected via Streamlit's `st.markdown()`. Modify the style section in `app.py` to customize:

- Color scheme (blue gradient in header)
- Message bubble styles
- Source document styling
- Font sizes and spacing

### Features

You can extend the app with:
- Export conversation history
- Customize the system prompt
- Add user preferences/settings
- Implement conversation ratings

## Interview Tips

This codebase is designed to be interview-friendly:

- **Clear structure**: Separation of concerns (API communication, UI rendering, state management)
- **Well-documented**: Comments explain key functionality
- **Modular functions**: Each function has a single responsibility
- **Error handling**: Graceful handling of API failures
- **Professional UI**: Demonstrates full-stack RAG implementation

## Troubleshooting

### "Cannot connect to API"
- Ensure the RAG API server is running on the configured URL
- Check that `API_BASE_URL` in `.env` is correct
- Verify the API is accessible: `curl http://localhost:8000/health`

### Responses are slow
- The RAG pipeline may be processing or retrieving documents
- Check API logs for bottlenecks
- Verify Pinecone/vector DB is responsive

### Session ID not persisting
- Clear browser cookies and cache
- Streamlit session state is per-browser session
- Closing and reopening the browser starts a new session

## License

MIT
