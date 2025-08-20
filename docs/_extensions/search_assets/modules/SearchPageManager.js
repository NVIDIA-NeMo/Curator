/**
 * Search Page Manager Module
 * Handles search functionality on the dedicated search page with filtering and grouping
 */

class SearchPageManager {
    constructor() {
        this.searchInput = null;
        this.resultsContainer = null;
        this.searchEngine = null;
        this.documents = [];
        this.currentQuery = '';
        this.allResults = [];
        this.currentFilters = {
            category: '',
            tag: '',
            type: '',
            persona: '',
            difficulty: '',
            modality: ''
        };
        this.filterOptions = {
            categories: [],
            tags: [],
            documentTypes: [],
            personas: [],
            difficulties: [],
            modalities: []
        };
        
        this.init();
    }
    
    async init() {
        console.log('🔍 Initializing search page...');
        
        // Get page elements
        this.searchInput = document.querySelector('#enhanced-search-page-input');
        this.resultsContainer = document.querySelector('#enhanced-search-page-results');
        
        if (!this.searchInput || !this.resultsContainer) {
            console.error('❌ Required search page elements not found');
            return;
        }
        
        // Wait for enhanced search to be available
        await this.waitForEnhancedSearch();
        
        // Create filter interface
        this.createFilterInterface();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Handle URL search parameter
        this.handleUrlSearch();
        
        console.log('✅ Search page initialized');
    }
    
    async waitForEnhancedSearch() {
        return new Promise((resolve) => {
            const checkForSearch = () => {
                if (window.enhancedSearchInstance && window.enhancedSearchInstance.isLoaded) {
                    this.searchEngine = window.enhancedSearchInstance.getSearchEngine();
                    this.documents = window.enhancedSearchInstance.getDocuments();
                    
                    // Get filter options
                    if (this.searchEngine && this.searchEngine.getFilterOptions) {
                        this.filterOptions = this.searchEngine.getFilterOptions();
                        console.log('✅ Filter options loaded:', this.filterOptions);
                    }
                    
                    resolve();
                } else {
                    setTimeout(checkForSearch, 100);
                }
            };
            checkForSearch();
        });
    }
    
    createFilterInterface() {
        // Get the search controls container
        const searchControlsContainer = this.searchInput.parentNode;
        
        // Add unified styling to the container
        searchControlsContainer.className = 'search-controls-container mb-4';
        
        // Create filter section
        const filterSection = document.createElement('div');
        filterSection.className = 'search-filters';
        filterSection.innerHTML = this.renderFilterInterface();
        
        // Insert filters before the search input within the same container
        searchControlsContainer.insertBefore(filterSection, this.searchInput);
        
        // Add search input wrapper class for consistent styling
        this.searchInput.className = 'form-control search-input-unified';
        
        // Bind filter events
        this.bindFilterEvents();
    }
    
    renderFilterInterface() {
        const categoryOptions = this.filterOptions.categories.map(cat => 
            `<option value="${cat}">${this.formatCategoryName(cat)}</option>`
        ).join('');
        
        const tagOptions = this.filterOptions.tags.map(tag => 
            `<option value="${tag}">${tag}</option>`
        ).join('');
        
        const typeOptions = this.filterOptions.documentTypes.map(type => 
            `<option value="${type}">${this.formatTypeName(type)}</option>`
        ).join('');
        
        const personaOptions = this.filterOptions.personas.map(persona => 
            `<option value="${persona}">${this.formatPersonaName(persona)}</option>`
        ).join('');
        
        const difficultyOptions = this.filterOptions.difficulties.map(difficulty => 
            `<option value="${difficulty}">${this.formatDifficultyName(difficulty)}</option>`
        ).join('');
        
        const modalityOptions = this.filterOptions.modalities.map(modality => 
            `<option value="${modality}">${this.formatModalityName(modality)}</option>`
        ).join('');
        
        return `
            <div class="filter-row">
                <div class="filter-group">
                    <select id="category-filter" class="filter-select">
                        <option value="">All Categories</option>
                        ${categoryOptions}
                    </select>
                </div>
                
                <div class="filter-group">
                    <select id="tag-filter" class="filter-select">
                        <option value="">All Tags</option>
                        ${tagOptions}
                    </select>
                </div>
                
                <div class="filter-group">
                    <select id="type-filter" class="filter-select">
                        <option value="">All Types</option>
                        ${typeOptions}
                    </select>
                </div>
                
                <div class="filter-actions">
                    <button id="clear-filters" class="btn btn-secondary btn-sm">
                        <i class="fa-solid fa-xmark"></i> Clear
                    </button>
                </div>
            </div>
        `;
    }
    
    formatCategoryName(category) {
        return category
            .split(/[-_]/)
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    formatTypeName(type) {
        return type
            .split(/[-_]/)
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    formatPersonaName(persona) {
        // Convert "data-scientist-focused" to "Data Scientist Focused"
        return persona
            .replace(/-focused$/, '') // Remove "-focused" suffix
            .split(/[-_]/)
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    formatDifficultyName(difficulty) {
        return difficulty.charAt(0).toUpperCase() + difficulty.slice(1);
    }
    
    formatModalityName(modality) {
        return modality
            .split(/[-_]/)
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    bindFilterEvents() {
        // Category filter
        document.getElementById('category-filter').addEventListener('change', (e) => {
            this.currentFilters.category = e.target.value;
            this.applyFiltersAndSearch();
        });
        
        // Tag filter
        document.getElementById('tag-filter').addEventListener('change', (e) => {
            this.currentFilters.tag = e.target.value;
            this.applyFiltersAndSearch();
        });
        
        // Type filter
        document.getElementById('type-filter').addEventListener('change', (e) => {
            this.currentFilters.type = e.target.value;
            this.applyFiltersAndSearch();
        });
        
        // Clear filters
        document.getElementById('clear-filters').addEventListener('click', () => {
            this.clearFilters();
        });
    }
    
    clearFilters() {
        this.currentFilters = { 
            category: '', 
            tag: '', 
            type: '',
            persona: '',
            difficulty: '',
            modality: ''
        };
        
        // Reset filter selects
        document.getElementById('category-filter').value = '';
        document.getElementById('tag-filter').value = '';
        document.getElementById('type-filter').value = '';
        
        // Clear active filter display
        this.updateActiveFiltersDisplay();
        
        // Re-run search
        this.applyFiltersAndSearch();
    }
    
    handleBadgeClick(filterType, filterValue) {
        // Update the appropriate filter
        this.currentFilters[filterType] = filterValue;
        
        // Update dropdown if it exists
        const dropdown = document.getElementById(`${filterType}-filter`);
        if (dropdown) {
            dropdown.value = filterValue;
        }
        
        // Update active filters display
        this.updateActiveFiltersDisplay();
        
        // Re-run search
        this.applyFiltersAndSearch();
    }
    
    updateActiveFiltersDisplay() {
        // Remove existing active filters display
        const existingDisplay = document.querySelector('.active-filters-display');
        if (existingDisplay) {
            existingDisplay.remove();
        }
        
        // Check for active metadata filters (not in dropdowns)
        const activeMetadataFilters = [];
        if (this.currentFilters.persona) {
            activeMetadataFilters.push(`👤 ${this.formatPersonaName(this.currentFilters.persona)}`);
        }
        if (this.currentFilters.difficulty) {
            activeMetadataFilters.push(`${this.getDifficultyIcon(this.currentFilters.difficulty)} ${this.formatDifficultyName(this.currentFilters.difficulty)}`);
        }
        if (this.currentFilters.modality) {
            activeMetadataFilters.push(`${this.getModalityIcon(this.currentFilters.modality)} ${this.formatModalityName(this.currentFilters.modality)}`);
        }
        
        if (activeMetadataFilters.length > 0) {
            const filtersContainer = document.querySelector('.search-filters');
            const activeFiltersHtml = `
                <div class="active-filters-display mb-2">
                    <small class="text-muted">Active filters: </small>
                    ${activeMetadataFilters.map(filter => `<span class="active-filter-badge">${filter}</span>`).join(' ')}
                    <button class="btn btn-outline-secondary btn-sm ms-2" onclick="window.searchPageManager.clearMetadataFilters()">
                        <i class="fa-solid fa-xmark"></i> Clear metadata filters
                    </button>
                </div>
            `;
            filtersContainer.insertAdjacentHTML('afterend', activeFiltersHtml);
        }
    }
    
    clearMetadataFilters() {
        this.currentFilters.persona = '';
        this.currentFilters.difficulty = '';
        this.currentFilters.modality = '';
        this.updateActiveFiltersDisplay();
        this.applyFiltersAndSearch();
    }
    
    applyFiltersAndSearch() {
        if (this.currentQuery) {
            this.handleSearch(this.currentQuery);
        }
    }
    
    setupEventListeners() {
        // Search input
        this.searchInput.addEventListener('input', this.debounce((e) => {
            this.handleSearch(e.target.value);
        }, 300));
        
        this.searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.handleSearch(e.target.value);
            }
        });
        
        // Badge click handlers (using event delegation)
        this.resultsContainer.addEventListener('click', (e) => {
            if (e.target.classList.contains('clickable-badge')) {
                const filterType = e.target.dataset.filterType;
                const filterValue = e.target.dataset.filterValue;
                this.handleBadgeClick(filterType, filterValue);
            }
        });
        
        // Make instance available globally for button callbacks
        window.searchPageManager = this;
        
        // Focus input on page load
        this.searchInput.focus();
    }
    
    handleUrlSearch() {
        const urlParams = new URLSearchParams(window.location.search);
        const query = urlParams.get('q');
        if (query) {
            this.searchInput.value = query;
            this.handleSearch(query);
        }
    }
    
    handleSearch(query) {
        this.currentQuery = query.trim();
        
        if (!this.currentQuery) {
            this.showEmptyState();
            return;
        }
        
        if (this.currentQuery.length < 2) {
            this.showMinLengthMessage();
            return;
        }
        
        // Perform search with filters
        const results = this.searchEngine.search(this.currentQuery, this.currentFilters);
        this.allResults = results;
        this.displayResults(results);
        
        // Update URL without reload
        const newUrl = new URL(window.location);
        newUrl.searchParams.set('q', this.currentQuery);
        window.history.replaceState(null, '', newUrl);
    }
    
        displayResults(results) {
        if (results.length === 0) {
            this.showNoResults();
            return;
        }

        const resultsHtml = results.map((result, index) => this.renderResult(result, index)).join('');
        
        this.resultsContainer.innerHTML = `
            <div id="ai-assistant-container" class="ai-assistant-container mb-4" style="display: none;"></div>
            <div class="search-results-header mb-4">
                <h3>Search Results</h3>
                <p class="text-muted">
                    Found ${results.length} result${results.length !== 1 ? 's' : ''} for "${this.escapeHtml(this.currentQuery)}"
                    ${this.getActiveFiltersText()}
                </p>
            </div>
            <div class="search-results-list">
                ${resultsHtml}
            </div>
        `;
        
        // Emit event for AI assistant integration
        this.emitSearchAIRequest(this.currentQuery, results);
    }
    
    getActiveFiltersText() {
        const activeFilters = [];
        
        if (this.currentFilters.category) {
            activeFilters.push(`Category: ${this.formatCategoryName(this.currentFilters.category)}`);
        }
        if (this.currentFilters.tag) {
            activeFilters.push(`Tag: ${this.currentFilters.tag}`);
        }
        if (this.currentFilters.type) {
            activeFilters.push(`Type: ${this.formatTypeName(this.currentFilters.type)}`);
        }
        if (this.currentFilters.persona) {
            activeFilters.push(`Persona: ${this.formatPersonaName(this.currentFilters.persona)}`);
        }
        if (this.currentFilters.difficulty) {
            activeFilters.push(`Difficulty: ${this.formatDifficultyName(this.currentFilters.difficulty)}`);
        }
        if (this.currentFilters.modality) {
            activeFilters.push(`Modality: ${this.formatModalityName(this.currentFilters.modality)}`);
        }
        
        return activeFilters.length > 0 ? ` (filtered by ${activeFilters.join(', ')})` : '';
    }
    
    renderResult(result, index) {
        const title = this.highlightText(result.title, this.currentQuery);
        const summary = this.highlightText(result.content?.substring(0, 200) || result.summary || '', this.currentQuery);
        const breadcrumb = this.getBreadcrumb(result.id);
        const sectionInfo = this.getSectionInfo(result.id);
        const matchingSections = this.renderMatchingSections(result, this.currentQuery);
        const resultTags = this.renderResultTags(result);
        const resultCategories = this.renderResultCategories(result);
        const metadataBadges = this.renderMetadataBadges(result);
        
        // Multiple matches indicator
        const multipleMatchesIndicator = result.totalMatches > 1 
            ? `<span class="multiple-matches-indicator">+${result.totalMatches - 1} more matches</span>`
            : '';
        
        return `
            <div class="search-result mb-4">
                <div class="result-header d-flex align-items-start mb-2">
                    <div class="section-icon me-3">
                        <i class="${sectionInfo.icon}"></i>
                    </div>
                    <div class="result-info flex-grow-1">
                        <h4 class="result-title mb-1">
                            <a href="${this.getDocumentUrl(result)}" class="text-decoration-none">${title}</a>
                            ${multipleMatchesIndicator}
                        </h4>
                        <div class="result-breadcrumb mb-2">
                            <small class="text-muted">${breadcrumb}</small>
                        </div>
                        <div class="result-meta d-flex align-items-center gap-2 mb-2 flex-wrap">
                            ${metadataBadges}
                        </div>
                        ${resultTags}
                    </div>
                </div>
                <div class="result-content">
                    <p class="result-summary mb-3">${summary}${summary.length >= 200 ? '...' : ''}</p>
                    ${matchingSections}
                </div>
            </div>
        `;
    }
    
    renderResultTags(result) {
        const tags = this.searchEngine.getDocumentTags(result);
        if (!tags || tags.length === 0) return '';
        
        const tagsToShow = tags.slice(0, 6); // Show more tags since they're now on their own line
        const tagsHtml = tagsToShow.map(tag => 
            `<span class="result-tag clickable-badge" data-filter-type="tag" data-filter-value="${this.escapeHtml(tag)}" title="Click to filter by this tag">${tag}</span>`
        ).join('');
        
        const moreText = tags.length > 6 ? `<span class="more-tags">+${tags.length - 6} more</span>` : '';
        
        return `<div class="result-tags mb-2">${tagsHtml}${moreText}</div>`;
    }
    
    renderResultCategories(result) {
        const categories = this.searchEngine.getDocumentCategories(result);
        if (!categories || categories.length === 0) return '';
        
        const categoriesHtml = categories.slice(0, 2).map(category => 
            `<span class="result-category badge bg-info">${this.formatCategoryName(category)}</span>`
        ).join('');
        
        return `<div class="result-categories">${categoriesHtml}</div>`;
    }
    
    renderMetadataBadges(result) {
        const badges = [];
        
        // Persona badge
        if (result.personas) {
            const personas = Array.isArray(result.personas) ? result.personas : [result.personas];
            const firstPersona = personas[0]; // Use first persona for filtering
            const personaText = personas.map(p => this.formatPersonaName(p)).join(', ');
            badges.push(`<span class="metadata-badge persona-badge clickable-badge" data-filter-type="persona" data-filter-value="${this.escapeHtml(firstPersona)}" title="Click to filter by ${this.formatPersonaName(firstPersona)}">👤 ${personaText}</span>`);
        }
        
        // Difficulty badge
        if (result.difficulty) {
            const difficultyIcon = this.getDifficultyIcon(result.difficulty);
            badges.push(`<span class="metadata-badge difficulty-badge clickable-badge" data-filter-type="difficulty" data-filter-value="${this.escapeHtml(result.difficulty)}" title="Click to filter by ${this.formatDifficultyName(result.difficulty)}">${difficultyIcon} ${this.formatDifficultyName(result.difficulty)}</span>`);
        }
        
        // Modality badge
        if (result.modality) {
            const modalityIcon = this.getModalityIcon(result.modality);
            badges.push(`<span class="metadata-badge modality-badge clickable-badge" data-filter-type="modality" data-filter-value="${this.escapeHtml(result.modality)}" title="Click to filter by ${this.formatModalityName(result.modality)}">${modalityIcon} ${this.formatModalityName(result.modality)}</span>`);
        }
        
        return badges.join('');
    }
    
    getDifficultyIcon(difficulty) {
        switch (difficulty.toLowerCase()) {
            case 'beginner': return '🔰';
            case 'intermediate': return '📊';
            case 'advanced': return '🚀';
            case 'reference': return '📚';
            default: return '📖';
        }
    }
    
    getModalityIcon(modality) {
        switch (modality.toLowerCase()) {
            case 'text-only': return '📝';
            case 'image-only': return '🖼️';
            case 'video-only': return '🎥';
            case 'multimodal': return '🔀';
            case 'universal': return '🌐';
            default: return '📄';
        }
    }
    
    renderMatchingSections(result, query) {
        if (!result.matchingSections || result.matchingSections.length <= 1) {
            return '';
        }
        
        const sectionsToShow = result.matchingSections.slice(0, 5);
        const hasMore = result.matchingSections.length > 5;
        
        const sectionsHtml = sectionsToShow.map(section => {
            const sectionIcon = this.getSectionIcon(section.type, section.level);
            const sectionText = this.highlightText(section.text, query);
            const anchor = section.anchor ? `#${section.anchor}` : '';
            const sectionUrl = this.getDocumentUrl(result) + anchor;
            
            return `
                <a href="${sectionUrl}" class="section-link d-flex align-items-center text-decoration-none mb-1 p-2 rounded">
                    <span class="section-icon me-2">${sectionIcon}</span>
                    <span class="section-text flex-grow-1">${sectionText}</span>
                    <i class="fas fa-external-link-alt ms-2" style="font-size: 0.75rem;"></i>
                </a>
            `;
        }).join('');
        
        const moreIndicator = hasMore ? `
            <div class="text-muted small mt-1 ms-4">
                <i class="fas fa-ellipsis-h me-1"></i>
                +${result.matchingSections.length - 5} more sections
            </div>
        ` : '';
        
        return `
            <div class="matching-sections">
                <h5 class="h6 mb-2">
                    <i class="fas fa-list-ul me-1"></i>
                    Matching sections:
                </h5>
                <div class="section-links border rounded p-2">
                    ${sectionsHtml}
                    ${moreIndicator}
                </div>
            </div>
        `;
    }
    
    getSectionIcon(type, level) {
        switch (type) {
            case 'title':
                return '<i class="fas fa-file-lines"></i>';
            case 'heading':
                if (level <= 2) return '<i class="fas fa-heading"></i>';
                if (level <= 4) return '<i class="fas fa-heading text-muted"></i>';
                return '<i class="fas fa-heading text-muted"></i>';
            case 'content':
                return '<i class="fas fa-align-left text-muted"></i>';
            default:
                return '<i class="fas fa-circle-dot text-muted"></i>';
        }
    }
    
    getBreadcrumb(docId) {
        const parts = docId.split('/').filter(part => part && part !== 'index');
        return parts.length > 0 ? parts.join(' › ') : 'Home';
    }
    
    getSectionInfo(docId) {
        const path = docId.toLowerCase();
        
        if (path.includes('get-started') || path.includes('getting-started')) {
            return {
                class: 'getting-started',
                icon: 'fas fa-rocket',
                label: 'Getting Started'
            };
        } else if (path.includes('admin')) {
            return {
                class: 'admin',
                icon: 'fas fa-cog',
                label: 'Administration'
            };
        } else if (path.includes('reference') || path.includes('api')) {
            return {
                class: 'reference',
                icon: 'fas fa-book',
                label: 'Reference'
            };
        } else if (path.includes('about') || path.includes('concepts')) {
            return {
                class: 'about',
                icon: 'fas fa-info-circle',
                label: 'About'
            };
        } else if (path.includes('tutorial')) {
            return {
                class: 'tutorial',
                icon: 'fas fa-graduation-cap',
                label: 'Tutorial'
            };
        } else {
            return {
                class: 'default',
                icon: 'fas fa-file-lines',
                label: 'Documentation'
            };
        }
    }
    
    getDocumentUrl(result) {
        if (result.url) {
            return result.url;
        }
        return `${result.id.replace(/^\/+/, '')}.html`;
    }
    
    highlightText(text, query) {
        if (!query) return this.escapeHtml(text);
        
        const terms = query.toLowerCase().split(/\s+/).filter(term => term.length > 1);
        let highlightedText = this.escapeHtml(text);
        
        terms.forEach(term => {
            const regex = new RegExp(`(${this.escapeRegex(term)})`, 'gi');
            highlightedText = highlightedText.replace(regex, '<mark class="search-highlight">$1</mark>');
        });
        
        return highlightedText;
    }
    
    showEmptyState() {
        this.resultsContainer.innerHTML = `
            <div class="text-center py-4">
                <i class="fas fa-search fa-2x mb-3 text-success"></i>
                <h4>Search Documentation</h4>
                <p class="text-muted">Start typing to search across all documentation pages...</p>
                <div class="mt-3">
                    <small class="text-muted">
                        <i class="fas fa-lightbulb text-success"></i>
                        <strong>Search Tips:</strong> Use specific terms for better results • Use filters to narrow down results • Search includes titles, content, and headings
                    </small>
                </div>
            </div>
        `;
    }
    
    showMinLengthMessage() {
        this.resultsContainer.innerHTML = `
            <div class="text-center py-4">
                <i class="fas fa-keyboard fa-2x mb-3 text-muted"></i>
                <h4>Keep typing...</h4>
                <p class="text-muted">Enter at least 2 characters to search</p>
            </div>
        `;
    }
    
    showNoResults() {
        const filtersActive = this.currentFilters.category || this.currentFilters.tag || this.currentFilters.type;
        const suggestionText = filtersActive 
            ? 'Try clearing some filters or using different keywords'
            : 'Try different keywords or check your spelling';
        
        this.resultsContainer.innerHTML = `
            <div class="no-results text-center py-4">
                <i class="fas fa-search fa-2x mb-3 text-muted"></i>
                <h4>No results found</h4>
                <p class="text-muted">No results found for "${this.escapeHtml(this.currentQuery)}"${this.getActiveFiltersText()}</p>
                <div class="mt-3">
                    <small class="text-muted">
                        ${suggestionText}
                    </small>
                </div>
                ${filtersActive ? `
                    <div class="mt-3">
                        <button onclick="document.getElementById('clear-filters').click()" class="btn btn-outline-secondary btn-sm">
                            <i class="fas fa-times"></i> Clear Filters
                        </button>
                    </div>
                ` : ''}
            </div>
        `;
    }
    
    // Utility methods
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
    
    emitSearchAIRequest(query, results) {
        // Emit event for AI assistant integration (search page)
        const aiRequestEvent = new CustomEvent('search-ai-request', {
            detail: {
                query: query,
                results: results,
                count: results.length,
                container: 'ai-assistant-container'
            }
        });
        document.dispatchEvent(aiRequestEvent);
        
        console.log(`🤖 Emitted search-ai-request event for query: "${query}" with ${results.length} results`);
    }
} 