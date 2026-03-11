/**
 * DocumentLoader Module
 * Handles loading and managing search documents from JSON index
 */

class DocumentLoader {
    constructor() {
        this.documents = {};
        this.isLoaded = false;
    }
    
    /**
     * Load documents from JSON index files
     */
    async loadDocuments() {
        try {
            const data = await this.fetchDocumentData();
            this.processDocuments(data);
            this.isLoaded = true;
            console.log(`✅ Document loader initialized with ${Object.keys(this.documents).length} documents`);
        } catch (error) {
            console.error('Failed to load search documents:', error);
            throw error;
        }
    }
    
    /**
     * Fetch document data from various possible paths
     */
    async fetchDocumentData() {
        // Try different paths to account for different page depths
        const possiblePaths = [
            './index.json',
            '../index.json', 
            '../../index.json',
            '../../../index.json'
        ];
        
        for (const path of possiblePaths) {
            try {
                const response = await fetch(path);
                if (response.ok) {
                    const data = await response.json();
                    console.log(`✅ Loaded search index from: ${path}`);
                    return data;
                }
            } catch (error) {
                console.log(`❌ Failed to load from ${path}: ${error.message}`);
            }
        }
        
        throw new Error('Failed to load search data from any path');
    }
    
    /**
     * Process and filter documents from raw data
     */
    processDocuments(data) {
        const allDocs = data.children || [data]; // Handle both formats
        
        // Filter out problematic documents
        const filteredDocs = allDocs.filter(doc => this.isValidDocument(doc));
        
        // Store documents by ID
        filteredDocs.forEach(doc => {
            this.documents[doc.id] = this.sanitizeDocument(doc);
        });
        
        console.log(`Processed ${filteredDocs.length} documents (filtered from ${allDocs.length} total)`);
    }
    
    /**
     * Check if a document is valid for indexing
     */
    isValidDocument(doc) {
        const docId = doc.id || '';
        return !docId.toLowerCase().includes('readme') && 
               !docId.startsWith('_') && 
               doc.title && 
               doc.content;
    }
    
    /**
     * Sanitize document content for safe indexing
     */
    sanitizeDocument(doc) {
        return {
            ...doc,
            title: this.sanitizeText(doc.title, 200),
            content: this.sanitizeText(doc.content, 5000),
            summary: this.sanitizeText(doc.summary, 500),
            headings: this.sanitizeHeadings(doc.headings),
            headings_text: this.sanitizeText(doc.headings_text, 1000),
            keywords: this.sanitizeArray(doc.keywords, 300),
            tags: this.sanitizeArray(doc.tags, 200),
            categories: this.sanitizeArray(doc.categories, 200),
            doc_type: this.sanitizeText(doc.doc_type, 50),
            section_path: this.sanitizeArray(doc.section_path, 200),
            author: this.sanitizeText(doc.author, 100)
        };
    }
    
    /**
     * Sanitize text content with length limits
     */
    sanitizeText(text, maxLength) {
        if (!text || typeof text !== 'string') return '';
        return text.substring(0, maxLength);
    }
    
    /**
     * Sanitize array content
     */
    sanitizeArray(arr, maxLength) {
        if (!Array.isArray(arr)) return [];
        return arr.map(item => String(item)).join(' ').substring(0, maxLength);
    }
    
    /**
     * Sanitize headings array
     */
    sanitizeHeadings(headings) {
        if (!Array.isArray(headings)) return [];
        return headings.map(heading => ({
            text: this.sanitizeText(heading.text, 200),
            level: Number(heading.level) || 1
        }));
    }
    
    /**
     * Get all loaded documents
     */
    getDocuments() {
        return this.documents;
    }
    
    /**
     * Get a specific document by ID
     */
    getDocument(id) {
        return this.documents[id];
    }
    
    /**
     * Get document count
     */
    getDocumentCount() {
        return Object.keys(this.documents).length;
    }
    
    /**
     * Check if documents are loaded
     */
    isReady() {
        return this.isLoaded && Object.keys(this.documents).length > 0;
    }
    
    /**
     * Get documents as array for indexing
     */
    getDocumentsArray() {
        return Object.values(this.documents);
    }
    
    /**
     * Filter documents by criteria
     */
    filterDocuments(filterFn) {
        return this.getDocumentsArray().filter(filterFn);
    }
    
    /**
     * Get document statistics
     */
    getStatistics() {
        const docs = this.getDocumentsArray();
        return {
            totalDocuments: docs.length,
            documentsWithSummary: docs.filter(d => d.summary).length,
            documentsWithHeadings: docs.filter(d => d.headings && d.headings.length > 0).length,
            documentsWithTags: docs.filter(d => d.tags && d.tags.length > 0).length,
            averageContentLength: docs.reduce((sum, d) => sum + (d.content?.length || 0), 0) / docs.length
        };
    }
}

// Make DocumentLoader available globally
window.DocumentLoader = DocumentLoader; 