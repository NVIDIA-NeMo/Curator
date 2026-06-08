# AI Assistant Extension

This Sphinx extension provides AI-powered analysis and responses for documentation search queries using external AI services.

## Features

- **AI-powered analysis** using external AI services (Pinecone Assistant API)
- **Smart triggering** based on search results count
- **Caching system** to reduce API calls
- **Configurable settings** for different AI providers
- **Graceful fallbacks** when AI services are unavailable
- **Usage statistics** tracking for API calls
- **NVIDIA theme integration** with proper styling

## Directory Structure

```
ai_assistant/
├── __init__.py                      # Extension setup & asset management
├── assets/
│   └── styles/
│       └── ai-assistant.css         # AI Assistant styling
├── core/
│   ├── main.js                     # Main coordinator & public API
│   ├── AIClient.js                 # API communication & core logic
│   └── ResponseProcessor.js        # Response processing & caching
├── ui/
│   ├── ResponseRenderer.js         # Response rendering & UI states
│   └── MarkdownProcessor.js        # Markdown to HTML conversion
├── integrations/
│   └── search-integration.js       # Search system integration
└── README.md                       # This file
```

## Modular Architecture

The AI Assistant extension uses a modular architecture for better maintainability and scalability:

### Core Modules

- **`main.js`**: Main coordinator that brings together all modules and provides unified API
- **`AIClient.js`**: Handles API communication with AI services and core analysis logic
- **`ResponseProcessor.js`**: Manages response processing, caching, and data transformation

### UI Modules

- **`ResponseRenderer.js`**: Handles all rendering methods (standard, error, loading, compact views)
- **`MarkdownProcessor.js`**: Converts markdown content to HTML with advanced features

### Integration Modules

- **`search-integration.js`**: Integrates with the enhanced search extension for seamless functionality

## What the Extension Does

1. **Modular Loading**: Dynamically loads modules with fallback path resolution
2. **AI Integration**: Connects to external AI services for intelligent query analysis
3. **Smart Caching**: Reduces API calls with intelligent response caching
4. **Flexible Rendering**: Supports multiple rendering formats (full, compact, summary)
5. **Asset Management**: Automatically includes CSS and JavaScript files with proper directory structure
6. **Build Integration**: Copies assets to `_static` preserving directory structure

## Usage

Add to your `conf.py`:

```python
extensions = [
    # ... other extensions
    "ai_assistant",  # AI Assistant extension
]

# Optional AI Assistant configuration
ai_assistant_enabled = True  # Enable/disable AI Assistant
ai_assistant_endpoint = "https://prod-1-data.ke.pinecone.io/assistant/chat/test-assistant"
ai_assistant_api_key = "your-api-key-here"
ai_trigger_threshold = 2  # Trigger AI when fewer than N results
ai_auto_trigger = True  # Auto-trigger AI analysis
```

## Configuration Options

- `ai_assistant_enabled`: Enable or disable the AI Assistant (default: True)
- `ai_assistant_endpoint`: API endpoint for the AI service
- `ai_assistant_api_key`: API key for authentication
- `ai_trigger_threshold`: Minimum search results to trigger AI (default: 2)
- `ai_auto_trigger`: Whether to automatically trigger AI analysis (default: True)

## Integration with Search

The AI Assistant extension is designed to work alongside the enhanced search extension:

1. **Separation of Concerns**: Search handles basic functionality, AI handles intelligent analysis
2. **Optional Integration**: AI can be disabled without affecting search functionality
3. **Shared Interface**: Both extensions can be used together seamlessly
4. **Context Enhancement**: Search results provide context for more accurate AI responses

## AI Service Integration

Currently supports:
- **Pinecone Assistant API**: RAG-powered responses using your documentation
- **Custom AI Services**: Extensible for other AI providers

### Pinecone Assistant Format

Expected request format:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "search query"
    }
  ],
  "stream": false,
  "model": "gpt-4o"
}
```

Expected response format:
```json
{
  "choices": [
    {
      "message": {
        "content": "AI response content"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  }
}
```

## JavaScript API

The extension provides a global `AIAssistant` class:

```javascript
// Create AI Assistant instance
const aiAssistant = new AIAssistant({
    enableAI: true,
    assistantApiKey: 'your-key',
    assistantEndpoint: 'https://api.endpoint.com',
    aiTriggerThreshold: 2,
    autoTrigger: true
});

// Analyze a query
const response = await aiAssistant.analyzeQuery('search query', searchResults);

// Render the AI response
const html = aiAssistant.renderResponse(response, 'search query');
```

## Asset Management

The extension uses an improved asset management pattern:

- **Organized Structure**: Assets are organized in logical directories
- **Automatic Copying**: Directory structure is preserved during build
- **Path Resolution**: Proper path resolution for CSS and JavaScript files
- **Error Handling**: Graceful handling of missing assets

## Styling

The extension includes comprehensive CSS styling that:
- Integrates with NVIDIA theme colors
- Supports dark mode
- Provides responsive design
- Includes accessibility features
- Handles print styles

## Dependencies

- Requires Internet connection for AI service calls
- No additional JavaScript dependencies
- Works with any AI service that accepts HTTP requests

## Error Handling

The extension gracefully handles:
- Network failures
- API rate limits
- Invalid responses
- Service unavailability
- Authentication errors

## Performance Considerations

- **Caching**: Reduces redundant API calls
- **Request throttling**: Prevents excessive requests during typing
- **Asynchronous loading**: Non-blocking AI analysis
- **Fallback UI**: Maintains functionality when AI is unavailable 