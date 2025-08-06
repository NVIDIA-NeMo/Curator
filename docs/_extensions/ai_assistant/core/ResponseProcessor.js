/**
 * Response Processor Module
 * Handles response processing, caching, and data transformation
 */

class ResponseProcessor {
    constructor() {
        this.cache = new Map();
        this.maxCacheSize = 50; // Limit cache size
    }
    
    normalizeQuery(query) {
        return query.toLowerCase().trim().replace(/\s+/g, ' ');
    }
    
    /**
     * Process and cache AI response
     */
    processResponse(query, rawResponse) {
        const normalizedQuery = this.normalizeQuery(query);
        
        // Check cache first
        if (this.cache.has(normalizedQuery)) {
            return { ...this.cache.get(normalizedQuery), cached: true };
        }
        
        // Handle error responses
        if (rawResponse.error) {
            return this.processError(new Error(rawResponse.message), query);
        }
        
        // Process the response
        const processedResponse = {
            content: rawResponse.content,
            usage: rawResponse.usage,
            cached: false,
            processedAt: new Date().toISOString()
        };
        
        // Cache the response
        this.cache.set(normalizedQuery, { ...processedResponse, cached: true });
        
        return processedResponse;
    }
    
    /**
     * Check if response is cached
     */
    hasCachedResponse(query) {
        const normalizedQuery = this.normalizeQuery(query);
        return this.cache.has(normalizedQuery);
    }
    
    /**
     * Get cached response
     */
    getCachedResponse(query) {
        const normalizedQuery = this.normalizeQuery(query);
        if (this.cache.has(normalizedQuery)) {
            const cached = this.cache.get(normalizedQuery);
            return {
                ...cached,
                cached: true,
                cacheTimestamp: cached.timestamp
            };
        }
        return null;
    }
    
    /**
     * Process error response
     */
    processError(error, query = '') {
        return {
            error: true,
            message: error.message || 'Unknown error occurred',
            content: null,
            query: query,
            processedAt: new Date().toISOString()
        };
    }
    
    /**
     * Validate response format
     */
    validateResponse(response) {
        if (!response) {
            throw new Error('No response received');
        }
        
        if (response.error) {
            return false; // Error responses are valid but indicate failure
        }
        
        if (!response.content || typeof response.content !== 'string') {
            throw new Error('Invalid response format: missing or invalid content');
        }
        
        return true;
    }
    
    /**
     * Extract usage statistics
     */
    extractUsageStats(response) {
        if (!response.usage) {
            return null;
        }
        
        const usage = response.usage;
        const totalTokens = usage.total_tokens || 
                          (usage.prompt_tokens || 0) + (usage.completion_tokens || 0);
        
        return {
            promptTokens: usage.prompt_tokens || 0,
            completionTokens: usage.completion_tokens || 0,
            totalTokens: totalTokens,
            hasUsageData: totalTokens > 0
        };
    }
    
    /**
     * Clear cache
     */
    clearCache() {
        this.cache.clear();
    }
    
    /**
     * Get cache size
     */
    getCacheSize() {
        return this.cache.size;
    }
    
    /**
     * Get cache keys (for debugging)
     */
    getCacheKeys() {
        return Array.from(this.cache.keys());
    }
    
    /**
     * Remove specific cache entry
     */
    removeCachedResponse(query) {
        const normalizedQuery = this.normalizeQuery(query);
        const removed = this.cache.delete(normalizedQuery);
        return removed;
    }
    
    /**
     * Get cache statistics
     */
    getCacheStats() {
        return {
            size: this.cache.size,
            keys: this.getCacheKeys(),
            memoryUsage: this.estimateMemoryUsage()
        };
    }
    
    /**
     * Estimate memory usage (rough calculation)
     */
    estimateMemoryUsage() {
        let totalSize = 0;
        for (const [key, value] of this.cache) {
            totalSize += key.length;
            totalSize += JSON.stringify(value).length;
        }
        return {
            bytes: totalSize,
            kb: Math.round(totalSize / 1024 * 100) / 100,
            entries: this.cache.size
        };
    }
    
    cacheResponse(normalizedQuery, response) {
        // Manage cache size
        if (this.cache.size >= this.maxCacheSize) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
        
        this.cache.set(normalizedQuery, {
            ...response,
            timestamp: new Date().toISOString()
        });
    }
}

// Make ResponseProcessor available globally
window.ResponseProcessor = ResponseProcessor; 