/**
 * AI Assistant Search Integration
 * Handles integration between search_assets and ai_assistant extensions
 */

// Global AI Assistant instance for search integration
let aiAssistantInstance = null;

// Initialize AI Assistant for search integration
function initializeAIAssistantForSearch() {
    if (typeof window.AIAssistant === 'undefined') {
        return;
    }
    
    // Create AI Assistant instance with search-optimized settings
    aiAssistantInstance = new window.AIAssistant({
        enableAI: true,
        aiTriggerThreshold: 2,
        autoTrigger: true,
        debounceDelay: 2000
    });
    
    // Make it globally available
    window.aiAssistantInstance = aiAssistantInstance;
    
    // Listen for search events
    document.addEventListener('search-ai-request', handleSearchAIRequest);
    document.addEventListener('enhanced-search-results', handleEnhancedSearchResults);
}

// Handle AI request from search page
async function handleSearchAIRequest(event) {
    const { query, results, count, container } = event.detail;
    
    const containerElement = document.getElementById(container);
    if (!containerElement) {
        return;
    }
    
    // Show loading state
    containerElement.style.display = 'block';
    containerElement.innerHTML = aiAssistantInstance.renderLoading();
    
    try {
        // Analyze with AI
        const aiResponse = await aiAssistantInstance.analyzeQuery(query, results);
        
        if (aiResponse && !aiResponse.error) {
            // Show AI response
            containerElement.innerHTML = aiAssistantInstance.renderResponse(aiResponse, query);
        } else {
            // Show error state
            containerElement.innerHTML = aiAssistantInstance.renderError(
                aiResponse?.message || 'AI analysis failed'
            );
        }
    } catch (error) {
        console.error('❌ AI analysis error:', error);
        containerElement.innerHTML = aiAssistantInstance.renderError(error.message);
    }
}

// Handle search results from enhanced search modal
async function handleEnhancedSearchResults(event) {
    const { query, results, count } = event.detail;
    
    if (!aiAssistantInstance || !aiAssistantInstance.isAvailable()) {
        return;
    }
    
    // Check if we should trigger AI analysis
    if (count >= aiAssistantInstance.options.aiTriggerThreshold && !aiAssistantInstance.options.autoTrigger) {
        return;
    }
    
    // Use the AI Assistant to analyze the query
    try {
        const aiResponse = await aiAssistantInstance.analyzeQuery(query, results);
        
        if (aiResponse && !aiResponse.error) {
            // Emit event that modal can listen to for AI enhancement
            const aiResultEvent = new CustomEvent('ai-analysis-complete', {
                detail: { query, aiResponse, results }
            });
            document.dispatchEvent(aiResultEvent);
        }
    } catch (error) {
        console.error('❌ Modal AI analysis failed:', error);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Small delay to ensure both extensions are loaded
    setTimeout(initializeAIAssistantForSearch, 100);
}); 