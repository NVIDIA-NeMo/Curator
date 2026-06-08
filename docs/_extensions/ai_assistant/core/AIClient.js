/**
 * AI Client Module
 * Handles API communication and core AI analysis logic
 */

class AIClient {
    constructor(options = {}) {
        this.options = {
            enableAI: options.enableAI !== false,
            assistantApiKey: options.assistantApiKey || 'pcsk_7SbfwC_5GFY9wxgTFAsKVkswEDjNVwX3L1ZzYUgD9rigQc5CVxtAnZ32ZLBQhfdzQW1hbH',
            assistantEndpoint: options.assistantEndpoint || 'https://prod-1-data.ke.pinecone.io/assistant/chat/test-assistant',
            aiTriggerThreshold: options.aiTriggerThreshold || 2,
            autoTrigger: options.autoTrigger !== false,
            debounceDelay: options.debounceDelay || 2000,
            ...options
        };
        
        this.loading = false;
        this.currentQuery = '';
        this.timeout = null;
    }
    
    /**
     * Analyze query with AI and return enhanced response
     */
    async analyzeQuery(query, searchResults = []) {
        if (!this.options.enableAI) {
            return null;
        }
        
        // Check if we should trigger AI based on results count
        if (searchResults.length >= this.options.aiTriggerThreshold && !this.options.autoTrigger) {
            return null;
        }
        
        this.currentQuery = query;
        
        try {
            this.loading = true;
            
            // Prepare enhanced query with context from search results
            const enhancedQuery = this.buildEnhancedQuery(query, searchResults);
            
            const response = await fetch(this.options.assistantEndpoint, {
                method: 'POST',
                headers: {
                    "Api-Key": this.options.assistantApiKey,
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    messages: [
                        {
                            role: "user",
                            content: enhancedQuery
                        }
                    ],
                    stream: false,
                    model: "gpt-4o"
                })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`AI API returned status ${response.status}: ${errorText}`);
            }
            
            const data = await response.json();
            
            // Extract the AI response content
            let aiAnswer = '';
            if (data.choices && data.choices[0] && data.choices[0].message) {
                aiAnswer = data.choices[0].message.content;
            } else if (data.message && data.message.content) {
                aiAnswer = data.message.content;
            } else {
                throw new Error('Unexpected response format from AI');
            }
            
            if (!aiAnswer) {
                throw new Error('No answer received from AI');
            }
            
            const aiResponse = {
                content: aiAnswer,
                usage: data.usage,
                cached: false
            };
            
            return aiResponse;
            
        } catch (error) {
            console.error('🤖 AI Client error:', error);
            return {
                error: true,
                message: error.message,
                content: null
            };
        } finally {
            this.loading = false;
        }
    }
    
    /**
     * Build enhanced query with document context
     */
    buildEnhancedQuery(query, searchResults) {
        if (!searchResults || searchResults.length === 0) {
            return `${query}

Please provide specific references to documentation sections that support your answer. When possible, mention the specific document titles or section headings that contain the relevant information.`;
        }
        
        // Build context from top search results
        const context = searchResults.slice(0, 3).map(result => {
            const doc = result.ref ? window.enhancedSearchInstance?.documents[result.ref] : result;
            if (!doc) return '';
            
            return `Document: "${doc.title || 'Untitled'}"
URL: ${this.getDocumentUrl(doc)}
Content: ${(doc.content || '').substring(0, 500)}...`;
        }).filter(ctx => ctx.length > 0).join('\n\n');
        
        return `${query}

Context from relevant documentation:
${context}

Please provide a comprehensive answer based on the documentation context above. When referencing information, please mention the specific document titles and include relevant URLs or section references. Format your response to clearly indicate which sources support each part of your answer.`;
    }
    
    /**
     * Get document URL from search result
     */
    getDocumentUrl(doc) {
        if (doc.url) return doc.url;
        if (doc.id) {
            // Convert document ID to URL
            const baseUrl = window.location.origin + window.location.pathname.replace(/\/[^\/]*$/, '/');
            return baseUrl + doc.id.replace(/\.rst$|\.md$/, '.html');
        }
        return '#';
    }
    
    /**
     * Schedule AI analysis with debouncing
     */
    scheduleAnalysis(query, searchResults = []) {
        // Clear any existing timeout
        if (this.timeout) {
            clearTimeout(this.timeout);
        }
        
        // Set delay for AI analysis
        this.timeout = setTimeout(() => {
            this.analyzeQuery(query, searchResults);
        }, this.options.debounceDelay);
    }
    
    /**
     * Check if AI is enabled and available
     */
    isAvailable() {
        return this.options.enableAI && this.options.assistantApiKey && this.options.assistantEndpoint;
    }
    
    /**
     * Get current loading state
     */
    isLoading() {
        return this.loading;
    }
    
    /**
     * Get current query
     */
    getCurrentQuery() {
        return this.currentQuery;
    }
    
    /**
     * Get options
     */
    getOptions() {
        return this.options;
    }
}

// Make AIClient available globally
window.AIClient = AIClient; 