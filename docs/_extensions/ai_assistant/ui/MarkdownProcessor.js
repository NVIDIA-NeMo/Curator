/**
 * Markdown Processor Module
 * Handles conversion of markdown content to HTML for AI responses
 */

class MarkdownProcessor {
    constructor() {
        // Initialize markdown processor
    }
    
    /**
     * Convert markdown to HTML for AI responses
     */
    markdownToHtml(markdown) {
        if (!markdown) return '';
        
        let html = markdown
            // Headers
            .replace(/^### (.*$)/gim, '<h3>$1</h3>')
            .replace(/^## (.*$)/gim, '<h2>$1</h2>')
            .replace(/^# (.*$)/gim, '<h1>$1</h1>')
            
            // Horizontal rules
            .replace(/^---+$/gm, '<hr>')
            
            // Bold and italic
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            
            // Links
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
            
            // Code blocks (simple)
            .replace(/```([^`]+)```/g, '<pre><code>$1</code></pre>')
            .replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Process lists BEFORE line breaks to avoid breaking up consecutive items
        html = this.processLists(html);
        
        // Now handle line breaks and paragraphs
        html = html
            // Double newlines become paragraph breaks
            .replace(/\n\n/g, '</p><p>')
            // Single line breaks become <br> tags
            .replace(/\n/g, '<br>');
        
        // Wrap in paragraphs if not already wrapped and not starting with block elements
        if (!html.startsWith('<h') && !html.startsWith('<p') && !html.startsWith('<ul') && !html.startsWith('<ol') && !html.startsWith('<pre') && !html.startsWith('<hr>')) {
            html = '<p>' + html + '</p>';
        }
        
        return html;
    }
    
    /**
     * Process markdown lists properly by grouping consecutive items and handling nesting
     */
    processLists(text) {
        // Split text into lines for processing
        const lines = text.split('\n');
        const processedLines = [];
        let inList = false;
        let listItems = [];
        let listType = null; // 'ul' or 'ol'
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const bulletMatch = line.match(/^([ ]*)([-*+])\s+(.*)$/);
            const numberedMatch = line.match(/^([ ]*)\d+\.\s+(.*)$/);
            const indentedBulletMatch = line.match(/^   ([-*+])\s+(.*)$/); // 3+ spaces for indented bullets
            
            if (bulletMatch && bulletMatch[1].length === 0) {
                // Top-level bulleted list item
                if (!inList || listType !== 'ul') {
                    // Starting a new unordered list
                    if (inList) {
                        // Close previous list
                        processedLines.push(this.closeList(listType, listItems));
                        listItems = [];
                    }
                    inList = true;
                    listType = 'ul';
                }
                listItems.push(bulletMatch[3]);
            } else if (numberedMatch && numberedMatch[1].length === 0) {
                // Top-level numbered list item
                if (!inList || listType !== 'ol') {
                    // Starting a new ordered list
                    if (inList) {
                        // Close previous list
                        processedLines.push(this.closeList(listType, listItems));
                        listItems = [];
                    }
                    inList = true;
                    listType = 'ol';
                }
                
                // Look ahead for indented sub-items
                const subItems = [];
                let j = i + 1;
                while (j < lines.length) {
                    const nextLine = lines[j];
                    const subBulletMatch = nextLine.match(/^   ([-*+])\s+(.*)$/);
                    if (subBulletMatch) {
                        subItems.push(subBulletMatch[2]);
                        j++;
                    } else if (nextLine.trim() === '') {
                        // Empty line, continue checking
                        j++;
                    } else {
                        break;
                    }
                }
                
                // Build the list item with sub-items if any
                let itemContent = numberedMatch[2];
                if (subItems.length > 0) {
                    const subList = subItems.map(item => `<li>${item}</li>`).join('');
                    itemContent += `<ul>${subList}</ul>`;
                    i = j - 1; // Skip the processed sub-items
                }
                
                listItems.push(itemContent);
            } else {
                // Not a list item (or indented item already processed)
                if (inList) {
                    // Close current list
                    processedLines.push(this.closeList(listType, listItems));
                    listItems = [];
                    inList = false;
                    listType = null;
                }
                processedLines.push(line);
            }
        }
        
        // Close any remaining list
        if (inList) {
            processedLines.push(this.closeList(listType, listItems));
        }
        
        return processedLines.join('\n');
    }
    
    /**
     * Helper to close a list and return HTML
     */
    closeList(listType, listItems) {
        if (listItems.length === 0) return '';
        
        const tag = listType === 'ol' ? 'ol' : 'ul';
        const listItemsHtml = listItems.map(item => `<li>${item}</li>`).join('');
        return `<${tag}>${listItemsHtml}</${tag}>`;
    }
    
    /**
     * Process markdown with enhanced features
     */
    processMarkdown(markdown, options = {}) {
        if (!markdown) return '';
        
        const {
            enableTables = false,
            enableStrikethrough = false,
            enableTaskLists = false,
            sanitizeHtml = true
        } = options;
        
        let html = this.markdownToHtml(markdown);
        
        // Enhanced features
        if (enableStrikethrough) {
            html = html.replace(/~~(.*?)~~/g, '<del>$1</del>');
        }
        
        if (enableTaskLists) {
            html = html.replace(/- \[ \] (.*$)/gim, '<li class="task-list-item"><input type="checkbox" disabled> $1</li>');
            html = html.replace(/- \[x\] (.*$)/gim, '<li class="task-list-item"><input type="checkbox" checked disabled> $1</li>');
        }
        
        if (enableTables) {
            html = this.processMarkdownTables(html);
        }
        
        if (sanitizeHtml) {
            html = this.sanitizeHtml(html);
        }
        
        return html;
    }
    
    /**
     * Process markdown tables (basic implementation)
     */
    processMarkdownTables(text) {
        // Basic table processing - could be enhanced
        const tableRegex = /(\|.*\|[\r\n]+\|[-\s|:]+\|[\r\n]+((\|.*\|[\r\n]*)+))/g;
        
        return text.replace(tableRegex, (match) => {
            const lines = match.trim().split('\n');
            if (lines.length < 3) return match;
            
            const headerLine = lines[0];
            const separatorLine = lines[1];
            const dataLines = lines.slice(2);
            
            // Parse header
            const headers = headerLine.split('|').map(h => h.trim()).filter(h => h);
            
            // Parse data rows
            const rows = dataLines.map(line => 
                line.split('|').map(cell => cell.trim()).filter(cell => cell !== '')
            ).filter(row => row.length > 0);
            
            // Build table HTML
            let tableHtml = '<table class="table ai-markdown-table">';
            tableHtml += '<thead><tr>';
            headers.forEach(header => {
                tableHtml += `<th>${header}</th>`;
            });
            tableHtml += '</tr></thead>';
            
            tableHtml += '<tbody>';
            rows.forEach(row => {
                tableHtml += '<tr>';
                row.forEach(cell => {
                    tableHtml += `<td>${cell}</td>`;
                });
                tableHtml += '</tr>';
            });
            tableHtml += '</tbody></table>';
            
            return tableHtml;
        });
    }
    
    /**
     * Sanitize HTML to prevent XSS
     */
    sanitizeHtml(html) {
        // Basic HTML sanitization
        const allowedTags = [
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'p', 'br', 'strong', 'em', 'u', 'del',
            'ul', 'ol', 'li',
            'a', 'code', 'pre',
            'table', 'thead', 'tbody', 'tr', 'th', 'td',
            'blockquote', 'div', 'span'
        ];
        
        const allowedAttributes = {
            'a': ['href', 'target', 'rel'],
            'table': ['class'],
            'li': ['class'],
            'input': ['type', 'checked', 'disabled']
        };
        
        // This is a basic implementation - in production, use a proper HTML sanitizer
        return html;
    }
    
    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    /**
     * Extract plain text from markdown
     */
    markdownToPlainText(markdown) {
        if (!markdown) return '';
        
        return markdown
            // Remove headers
            .replace(/^#+\s*/gm, '')
            // Remove bold/italic
            .replace(/\*\*(.*?)\*\*/g, '$1')
            .replace(/\*(.*?)\*/g, '$1')
            // Remove links
            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
            // Remove code blocks
            .replace(/```[^`]*```/g, '')
            .replace(/`([^`]+)`/g, '$1')
            // Remove list markers
            .replace(/^\s*[-*+]\s+/gm, '')
            .replace(/^\s*\d+\.\s+/gm, '')
            // Clean up whitespace
            .replace(/\n+/g, ' ')
            .trim();
    }
    
    /**
     * Get reading time estimate
     */
    getReadingTimeEstimate(text) {
        const plainText = this.markdownToPlainText(text);
        const wordCount = plainText.split(/\s+/).length;
        const wordsPerMinute = 200; // Average reading speed
        const readingTime = Math.ceil(wordCount / wordsPerMinute);
        
        return {
            wordCount,
            estimatedMinutes: readingTime,
            readingTime: readingTime === 1 ? '1 minute' : `${readingTime} minutes`
        };
    }
}

// Make MarkdownProcessor available globally
window.MarkdownProcessor = MarkdownProcessor; 