/**
 * SearchEngine Module
 * Handles Lunr.js integration and search logic with filtering and grouping
 */

class SearchEngine {
    constructor(utils) {
        this.utils = utils;
        this.index = null;
        this.documents = {};
        this.isInitialized = false;
        this.categories = new Set();
        this.tags = new Set();
        this.documentTypes = new Set();
        this.personas = new Set();
        this.difficulties = new Set();
        this.modalities = new Set();
    }
    
    /**
     * Initialize the search engine with documents
     */
    async initialize(documents) {
        try {
            await this.loadLunr();
            this.documents = documents;
            this.collectMetadata();
            this.buildIndex();
            this.isInitialized = true;
        } catch (error) {
            throw error;
        }
    }
    
    /**
     * Collect metadata for filtering (categories, tags, types) using actual frontmatter values
     */
    collectMetadata() {
        // Clear existing sets
        this.categories = new Set();
        this.tags = new Set();
        this.documentTypes = new Set();
        this.personas = new Set();
        this.difficulties = new Set();
        this.modalities = new Set();
        
        Object.values(this.documents).forEach(doc => {
            // Collect actual frontmatter categories (primary taxonomy)
            if (doc.categories) {
                if (Array.isArray(doc.categories)) {
                    doc.categories.forEach(cat => this.categories.add(cat));
                } else if (typeof doc.categories === 'string') {
                    doc.categories.split(',').forEach(cat => this.categories.add(cat.trim()));
                }
            }
            
            // Collect actual frontmatter tags
            if (doc.tags) {
                if (Array.isArray(doc.tags)) {
                    doc.tags.forEach(tag => {
                        // Split space-separated tags and add individually
                        if (typeof tag === 'string' && tag.includes(' ')) {
                            tag.split(' ').forEach(individualTag => {
                                if (individualTag.trim()) {
                                    this.tags.add(individualTag.trim());
                                }
                            });
                        } else if (tag && tag.trim()) {
                            this.tags.add(tag.trim());
                        }
                    });
                } else if (typeof doc.tags === 'string') {
                    // Handle both comma-separated and space-separated tags
                    const allTags = doc.tags.includes(',') 
                        ? doc.tags.split(',')
                        : doc.tags.split(' ');
                    
                    allTags.forEach(tag => {
                        if (tag && tag.trim()) {
                            this.tags.add(tag.trim());
                        }
                    });
                }
            }
            
            // Use actual content_type from frontmatter (not calculated doc_type)
            if (doc.content_type) {
                this.documentTypes.add(doc.content_type);
            }
            
            // Collect rich frontmatter taxonomy fields
            if (doc.personas) {
                if (Array.isArray(doc.personas)) {
                    doc.personas.forEach(persona => this.personas.add(persona));
                } else if (typeof doc.personas === 'string') {
                    this.personas.add(doc.personas);
                }
            }
            
            if (doc.difficulty) {
                this.difficulties.add(doc.difficulty);
            }
            
            if (doc.modality) {
                this.modalities.add(doc.modality);
            }
        });
    }
    
    /**
     * Get available filter options using actual frontmatter taxonomy
     */
    getFilterOptions() {
        return {
            categories: Array.from(this.categories).sort(),
            tags: Array.from(this.tags).sort(),
            documentTypes: Array.from(this.documentTypes).sort(),
            personas: Array.from(this.personas).sort(),
            difficulties: Array.from(this.difficulties).sort(),
            modalities: Array.from(this.modalities).sort()
        };
    }
    
    /**
     * Load Lunr.js library if not already loaded
     */
    async loadLunr() {
        if (typeof lunr === 'undefined') {
            await this.utils.loadScript('https://unpkg.com/lunr@2.3.9/lunr.min.js');
        }
    }
    
    /**
     * Build the Lunr search index
     */
    buildIndex() {
        const documentsArray = Object.values(this.documents);
        const self = this;
        
        this.index = lunr(function() {
            // Define fields with boosting
            this.ref('id');
            this.field('title', { boost: 10 });
            this.field('content', { boost: 5 });
            this.field('summary', { boost: 8 });
            this.field('headings', { boost: 6 });
            this.field('headings_text', { boost: 7 });
            this.field('keywords', { boost: 9 });
            this.field('tags', { boost: 4 });
            this.field('categories', { boost: 3 });
            this.field('content_type', { boost: 2 }); // Use actual frontmatter content_type
            this.field('personas', { boost: 3 }); // Add personas field
            this.field('difficulty', { boost: 2 }); // Add difficulty field
            this.field('modality', { boost: 2 }); // Add modality field
            this.field('section_path', { boost: 1 });
            this.field('author', { boost: 1 });
            
            // Add documents to index
            documentsArray.forEach((doc) => {
                try {
                    this.add({
                        id: doc.id,
                        title: doc.title || '',
                        content: (doc.content || '').substring(0, 5000), // Limit content length
                        summary: doc.summary || '',
                        headings: self.extractHeadingsText(doc.headings),
                        headings_text: doc.headings_text || '',
                        keywords: self.arrayToString(doc.keywords),
                        tags: self.arrayToString(doc.tags),
                        categories: self.arrayToString(doc.categories),
                        content_type: doc.content_type || '', // Use actual frontmatter content_type
                        personas: self.arrayToString(doc.personas), // Add actual frontmatter personas
                        difficulty: doc.difficulty || '', // Add actual frontmatter difficulty
                        modality: doc.modality || '', // Add actual frontmatter modality
                        section_path: self.arrayToString(doc.section_path),
                        author: doc.author || ''
                    });
                } catch (docError) {
                    // Skip documents that fail to index
                }
            }, this);
        });
    }
    
    /**
     * Convert array to string for indexing
     */
    arrayToString(arr) {
        if (Array.isArray(arr)) {
            return arr.join(' ');
        }
        return arr || '';
    }
    
    /**
     * Extract text from headings array
     */
    extractHeadingsText(headings) {
        if (!Array.isArray(headings)) return '';
        return headings.map(h => h.text || '').join(' ');
    }
    
    /**
     * Perform search with query and optional filters
     */
    search(query, filters = {}, maxResults = 20) {
        if (!this.isInitialized || !this.index) {
            return [];
        }
        
        if (!query || query.trim().length < 2) {
            return [];
        }
        
        try {
            // Enhanced search with multiple strategies
            const results = this.performMultiStrategySearch(query);
            
            // Process and enhance results
            const enhancedResults = this.enhanceResults(results, query);
            
            // Apply filters
            const filteredResults = this.applyFilters(enhancedResults, filters);
            
            // Group and rank results
            const groupedResults = this.groupResultsByDocument(filteredResults, query);
            
            return groupedResults.slice(0, maxResults);
                
        } catch (error) {
            return [];
        }
    }
    
    /**
     * Apply filters to search results
     */
    applyFilters(results, filters) {
        return results.filter(result => {
            // Category filter
            if (filters.category && filters.category !== '') {
                const docCategories = this.getDocumentCategories(result);
                if (!docCategories.includes(filters.category)) {
                    return false;
                }
            }
            
            // Tag filter
            if (filters.tag && filters.tag !== '') {
                const docTags = this.getDocumentTags(result);
                if (!docTags.includes(filters.tag)) {
                    return false;
                }
            }
            
            // Document type filter (using actual frontmatter content_type)
            if (filters.type && filters.type !== '') {
                if (result.content_type !== filters.type) {
                    return false;
                }
            }
            
            // Persona filter
            if (filters.persona && filters.persona !== '') {
                const docPersonas = this.getDocumentPersonas(result);
                if (!docPersonas.includes(filters.persona)) {
                    return false;
                }
            }
            
            // Difficulty filter
            if (filters.difficulty && filters.difficulty !== '') {
                if (result.difficulty !== filters.difficulty) {
                    return false;
                }
            }
            
            // Modality filter
            if (filters.modality && filters.modality !== '') {
                if (result.modality !== filters.modality) {
                    return false;
                }
            }
            
            return true;
        });
    }
    
    /**
     * Get categories for a document
     */
    getDocumentCategories(doc) {
        const categories = [];
        
        // From explicit categories
        if (doc.categories) {
            if (Array.isArray(doc.categories)) {
                categories.push(...doc.categories);
            } else {
                categories.push(...doc.categories.split(',').map(c => c.trim()));
            }
        }
        
        // From section path
        if (doc.section_path && Array.isArray(doc.section_path)) {
            categories.push(...doc.section_path);
        }
        
        // From document ID path
        if (doc.id) {
            const pathParts = doc.id.split('/').filter(part => part && part !== 'index');
            categories.push(...pathParts);
        }
        
        return [...new Set(categories)]; // Remove duplicates
    }
    
    /**
     * Get tags for a document
     */
    getDocumentTags(doc) {
        if (!doc.tags) return [];
        
        if (Array.isArray(doc.tags)) {
            // Handle array of tags that might contain space-separated strings
            const flatTags = [];
            doc.tags.forEach(tag => {
                if (typeof tag === 'string' && tag.includes(' ')) {
                    // Split space-separated tags
                    tag.split(' ').forEach(individualTag => {
                        if (individualTag.trim()) {
                            flatTags.push(individualTag.trim());
                        }
                    });
                } else if (tag && tag.trim()) {
                    flatTags.push(tag.trim());
                }
            });
            return flatTags;
        }
        
        // Handle string tags - check for both comma and space separation
        if (typeof doc.tags === 'string') {
            const allTags = [];
            const tagString = doc.tags.trim();
            
            if (tagString.includes(',')) {
                // Comma-separated tags
                tagString.split(',').forEach(tag => {
                    if (tag.trim()) {
                        allTags.push(tag.trim());
                    }
                });
            } else {
                // Space-separated tags
                tagString.split(' ').forEach(tag => {
                    if (tag.trim()) {
                        allTags.push(tag.trim());
                    }
                });
            }
            
            return allTags;
        }
        
        return [];
    }
    
    
    /**
     * Get personas for a document
     */
    getDocumentPersonas(doc) {
        if (!doc.personas) return [];
        
        if (Array.isArray(doc.personas)) {
            return doc.personas;
        }
        
        return [doc.personas];
    }
    
    /**
     * Perform search with multiple strategies
     */
    performMultiStrategySearch(query) {
        const strategies = [
            // Exact phrase search with wildcards
            `"${query}" ${query}*`,
            // Fuzzy search with wildcards  
            `${query}* ${query}~2`,
            // Individual terms with boost
            query.split(/\s+/).map(term => `${term}*`).join(' '),
            // Fallback: just the query
            query
        ];
        
        let allResults = [];
        const seenIds = new Set();
        
        for (const strategy of strategies) {
            try {
                const results = this.index.search(strategy);
                
                // Add new results (avoid duplicates)
                results.forEach(result => {
                    if (!seenIds.has(result.ref)) {
                        seenIds.add(result.ref);
                        allResults.push({
                            ...result,
                            strategy: strategy
                        });
                    }
                });
                
                // If we have enough good results, stop
                if (allResults.length >= 30) break;
                
            } catch (strategyError) {
                console.warn(`Search strategy failed: ${strategy}`, strategyError);
            }
        }
        
        return allResults;
    }
    
    /**
     * Enhance search results with document data
     */
    enhanceResults(results, query) {
        return results.map(result => {
            const doc = this.documents[result.ref];
            if (!doc) {
                console.warn(`Document not found: ${result.ref}`);
                return null;
            }
            
            return {
                ...doc,
                score: result.score,
                matchedTerms: Object.keys(result.matchData?.metadata || {}),
                matchData: result.matchData,
                strategy: result.strategy
            };
        }).filter(Boolean); // Remove null results
    }
    
    /**
     * Group results by document and find matching sections
     */
    groupResultsByDocument(results, query) {
        const grouped = new Map();
        
        results.forEach(result => {
            const docId = result.id;
            
            if (!grouped.has(docId)) {
                // Find matching sections within this document
                const matchingSections = this.findMatchingSections(result, query);
                
                grouped.set(docId, {
                    ...result,
                    matchingSections,
                    totalMatches: 1,
                    combinedScore: result.score
                });
            } else {
                // Document already exists, combine scores and sections
                const existing = grouped.get(docId);
                const additionalSections = this.findMatchingSections(result, query);
                
                existing.matchingSections = this.mergeSections(existing.matchingSections, additionalSections);
                existing.totalMatches += 1;
                existing.combinedScore = Math.max(existing.combinedScore, result.score);
            }
        });
        
        // Convert map to array and sort by combined score
        return Array.from(grouped.values())
            .sort((a, b) => b.combinedScore - a.combinedScore);
    }
    
    /**
     * Find matching sections within a document
     */
    findMatchingSections(result, query) {
        const matchingSections = [];
        const queryTerms = query.toLowerCase().split(/\s+/);
        
        // Check if title matches
        if (result.title) {
            const titleText = result.title.toLowerCase();
            const hasMatch = queryTerms.some(term => titleText.includes(term));
            
            if (hasMatch) {
                matchingSections.push({
                    type: 'title',
                    text: result.title,
                    level: 1,
                    anchor: ''
                });
            }
        }
        
        // Check headings for matches
        if (result.headings && Array.isArray(result.headings)) {
            result.headings.forEach(heading => {
                const headingText = heading.text?.toLowerCase() || '';
                const hasMatch = queryTerms.some(term => headingText.includes(term));
                
                if (hasMatch) {
                    matchingSections.push({
                        type: 'heading',
                        text: heading.text,
                        level: heading.level || 2,
                        anchor: this.generateAnchor(heading.text)
                    });
                }
            });
        }
        
        // If no specific sections found, add a general content match
        if (matchingSections.length === 0) {
            matchingSections.push({
                type: 'content',
                text: 'Content match',
                level: 0,
                anchor: ''
            });
        }
        
        return matchingSections;
    }
    
    /**
     * Generate anchor link similar to how Sphinx does it
     */
    generateAnchor(headingText) {
        if (!headingText) return '';
        
        return headingText
            .toLowerCase()
            .replace(/[^\w\s-]/g, '')  // Remove special chars
            .replace(/\s+/g, '-')      // Replace spaces with hyphens
            .trim();
    }
    
    /**
     * Merge sections, avoiding duplicates
     */
    mergeSections(existing, additional) {
        const merged = [...existing];
        
        additional.forEach(section => {
            const isDuplicate = existing.some(existingSection => 
                existingSection.text === section.text && 
                existingSection.type === section.type
            );
            
            if (!isDuplicate) {
                merged.push(section);
            }
        });
        
        return merged;
    }
    
    /**
     * Get search statistics
     */
    getStatistics() {
        return {
            documentsIndexed: Object.keys(this.documents).length,
            categoriesAvailable: this.categories.size,
            tagsAvailable: this.tags.size,
            documentTypesAvailable: this.documentTypes.size,
            isInitialized: this.isInitialized
        };
    }
    
    /**
     * Check if the search engine is ready
     */
    isReady() {
        return this.isInitialized && this.index !== null;
    }
}

// Make SearchEngine available globally
window.SearchEngine = SearchEngine; 