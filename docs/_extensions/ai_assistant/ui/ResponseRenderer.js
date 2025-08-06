/**
 * Response Renderer Module
 * Handles rendering of AI responses in various states and formats
 */

class ResponseRenderer {
    constructor(markdownProcessor = null) {
        this.markdownProcessor = markdownProcessor || new MarkdownProcessor();
    }
    
    /**
     * Render AI response in a standardized format
     */
    renderResponse(aiResponse, query) {
        if (!aiResponse || aiResponse.error) {
            return this.renderError(aiResponse?.message || 'AI Assistant unavailable');
        }
        
        // Calculate usage display
        const usageHTML = this.renderUsageStats(aiResponse.usage);
        
        const cacheIndicator = aiResponse.cached ? `
            <div class="mt-3 pt-3 border-top">
                <small class="ai-cache-indicator">
                    <i class="fa-solid fa-clock me-1"></i>
                    This response was cached from a previous query.
                </small>
            </div>
        ` : '';
        
        return `
            <div class="ai-assistant-response">
                <div class="ai-assistant-header">
                    <div class="d-flex align-items-center justify-content-between">
                        <div class="d-flex align-items-center">
                            <i class="fa-solid fa-robot ai-assistant-icon me-2"></i>
                            <h5 class="mb-0 ai-assistant-title">AI Analysis</h5>
                        </div>
                        <div class="d-flex align-items-center gap-2">
                            <span class="badge ai-status-badge ms-2">Complete</span>
                        </div>
                    </div>
                </div>
                <div class="ai-assistant-content">
                    <div class="ai-response-text">${this.markdownProcessor.markdownToHtml(aiResponse.content)}</div>
                    ${usageHTML}
                    ${cacheIndicator}
                </div>
                <div class="ai-assistant-footer">
                    <small class="ai-disclaimer">
                        This answer is generated from your documentation using AI. For the most accurate information, please refer to the original documents.
                    </small>
                </div>
            </div>
        `;
    }
    
    /**
     * Render usage statistics
     */
    renderUsageStats(usage) {
        if (!usage) return '';
        
        const totalTokens = usage.total_tokens || 
                          (usage.prompt_tokens || 0) + (usage.completion_tokens || 0);
        
        if (totalTokens <= 0) return '';
        
        return `
            <div class="ai-usage-stats mt-3 pt-3 border-top">
                <h6 class="ai-usage-title mb-2">
                    <i class="fa-solid fa-chart-bar me-1"></i>
                    Usage Statistics:
                </h6>
                <div class="row g-2">
                    <div class="col-md-4">
                        <div class="p-2 ai-usage-item rounded text-center">
                            <div class="fw-semibold ai-usage-number">${usage.prompt_tokens || 'N/A'}</div>
                            <small class="ai-usage-label">Input tokens</small>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="p-2 ai-usage-item rounded text-center">
                            <div class="fw-semibold ai-usage-number">${usage.completion_tokens || 'N/A'}</div>
                            <small class="ai-usage-label">Output tokens</small>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="p-2 ai-usage-item rounded text-center">
                            <div class="fw-semibold ai-usage-number">${totalTokens}</div>
                            <small class="ai-usage-label">Total tokens</small>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    /**
     * Render error state
     */
    renderError(message) {
        return `
            <div class="ai-assistant-response ai-assistant-error">
                <div class="ai-assistant-header">
                    <div class="d-flex align-items-center justify-content-between">
                        <div class="d-flex align-items-center">
                            <i class="fa-solid fa-robot ai-assistant-icon me-2"></i>
                            <h5 class="mb-0 ai-assistant-title">AI Analysis</h5>
                        </div>
                        <span class="badge ai-status-badge-error ms-2">Unavailable</span>
                    </div>
                </div>
                <div class="ai-assistant-content">
                    <div class="alert alert-info border-0 mb-0">
                        <div class="d-flex align-items-start">
                            <i class="fa-solid fa-info-circle ai-info-icon me-2 mt-1"></i>
                            <div>
                                <div class="fw-semibold ai-error-title">AI Analysis Currently Unavailable</div>
                                <div class="small ai-error-text mt-1">
                                    ${message || 'The AI assistant service is temporarily unavailable. Please check the search results below for relevant documentation.'}
                                </div>
                                <div class="small ai-error-suggestions mt-2">
                                    <strong>Possible solutions:</strong><br>
                                    • Make sure documents are uploaded to the assistant<br>
                                    • Check if the assistant configuration is correct<br>
                                    • Verify network connectivity
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    /**
     * Render loading state
     */
    renderLoading() {
        return `
            <div class="ai-assistant-response ai-assistant-loading">
                <div class="ai-assistant-header">
                    <div class="d-flex align-items-center justify-content-between">
                        <div class="d-flex align-items-center">
                            <i class="fa-solid fa-robot ai-assistant-icon me-2"></i>
                            <h5 class="mb-0 ai-assistant-title">AI Analysis</h5>
                        </div>
                        <span class="badge ai-status-badge-loading ms-2">Analyzing...</span>
                    </div>
                </div>
                <div class="ai-assistant-content">
                    <div class="d-flex align-items-center ai-loading-content">
                        <div class="spinner-border spinner-border-sm me-2 ai-spinner" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        Generating AI analysis of your query...
                    </div>
                </div>
            </div>
        `;
    }
    
    /**
     * Render manual trigger option
     */
    renderManualTrigger(onTrigger) {
        return `
            <div class="ai-assistant-response ai-assistant-manual">
                <div class="ai-assistant-header">
                    <div class="d-flex align-items-center justify-content-between">
                        <div class="d-flex align-items-center">
                            <i class="fa-solid fa-robot ai-assistant-icon me-2"></i>
                            <h5 class="mb-0 ai-assistant-title">AI Analysis</h5>
                        </div>
                        <button class="btn btn-sm btn-outline-success ai-manual-trigger">
                            <i class="fa-solid fa-play me-1"></i>
                            Analyze
                        </button>
                    </div>
                </div>
                <div class="ai-assistant-content">
                    <div class="text-center py-3 ai-manual-content">
                        <i class="fa-solid fa-robot fa-2x mb-2 ai-manual-icon"></i>
                        <p class="mb-0 ai-manual-text">AI analysis is available for this search.</p>
                        <p class="small mb-0 ai-manual-subtext">Click "Analyze" to get AI-powered insights.</p>
                    </div>
                </div>
            </div>
        `;
    }
    
    /**
     * Render compact response (for sidebars, modals, etc.)
     */
    renderCompactResponse(aiResponse, query) {
        if (!aiResponse || aiResponse.error) {
            return this.renderCompactError(aiResponse?.message || 'AI unavailable');
        }
        
        const shortContent = this.truncateContent(aiResponse.content, 200);
        const cacheIcon = aiResponse.cached ? '<i class="fa-solid fa-clock text-muted"></i>' : '';
        
        return `
            <div class="ai-assistant-compact">
                <div class="ai-compact-header d-flex align-items-center mb-2">
                    <i class="fa-solid fa-robot text-success me-2"></i>
                    <span class="fw-semibold">AI Insight</span>
                    ${cacheIcon}
                </div>
                <div class="ai-compact-content">
                    ${this.markdownProcessor.markdownToHtml(shortContent)}
                    ${shortContent.length < aiResponse.content.length ? '<small class="text-muted">...</small>' : ''}
                </div>
            </div>
        `;
    }
    
    /**
     * Render compact error
     */
    renderCompactError(message) {
        return `
            <div class="ai-assistant-compact ai-compact-error">
                <div class="ai-compact-header d-flex align-items-center mb-2">
                    <i class="fa-solid fa-robot text-muted me-2"></i>
                    <span class="fw-semibold text-muted">AI Unavailable</span>
                </div>
                <div class="ai-compact-content">
                    <small class="text-muted">${message || 'AI analysis failed'}</small>
                </div>
            </div>
        `;
    }
    
    /**
     * Render response summary (for search result enhancement)
     */
    renderResponseSummary(aiResponse, query) {
        if (!aiResponse || aiResponse.error) {
            return '';
        }
        
        const summary = this.extractSummary(aiResponse.content);
        
        return `
            <div class="ai-response-summary">
                <div class="ai-summary-header">
                    <i class="fa-solid fa-lightbulb text-warning me-1"></i>
                    <span class="fw-semibold">AI Insight:</span>
                </div>
                <div class="ai-summary-content">
                    ${this.markdownProcessor.markdownToHtml(summary)}
                </div>
            </div>
        `;
    }
    
    /**
     * Render debug information
     */
    renderDebugInfo(aiResponse, query, processingTime = null) {
        if (!aiResponse) return '';
        
        const debugData = {
            query: query,
            cached: aiResponse.cached || false,
            hasContent: !!aiResponse.content,
            hasUsage: !!aiResponse.usage,
            error: aiResponse.error || false,
            processingTime: processingTime,
            timestamp: new Date().toISOString()
        };
        
        return `
            <div class="ai-debug-info mt-3 p-3 bg-light rounded">
                <h6 class="mb-2">
                    <i class="fa-solid fa-bug me-1"></i>
                    Debug Information
                </h6>
                <pre class="small mb-0"><code>${JSON.stringify(debugData, null, 2)}</code></pre>
            </div>
        `;
    }
    
    /**
     * Truncate content for compact display
     */
    truncateContent(content, maxLength = 200) {
        if (!content || content.length <= maxLength) {
            return content;
        }
        
        // Try to truncate at word boundary
        const truncated = content.substring(0, maxLength);
        const lastSpace = truncated.lastIndexOf(' ');
        
        if (lastSpace > maxLength * 0.8) {
            return truncated.substring(0, lastSpace);
        }
        
        return truncated;
    }
    
    /**
     * Extract summary from AI response content
     */
    extractSummary(content, maxSentences = 2) {
        if (!content) return '';
        
        // Split into sentences (basic approach)
        const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 10);
        
        if (sentences.length <= maxSentences) {
            return content;
        }
        
        return sentences.slice(0, maxSentences).join('. ') + '.';
    }
    
    /**
     * Set markdown processor
     */
    setMarkdownProcessor(processor) {
        this.markdownProcessor = processor;
    }
    
    /**
     * Get markdown processor
     */
    getMarkdownProcessor() {
        return this.markdownProcessor;
    }
}

// Make ResponseRenderer available globally
window.ResponseRenderer = ResponseRenderer; 