// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

(args) => {
    const { scrollpos, debug } = args;
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;

    // scroll to random position
    const maxScrollY = document.documentElement.scrollHeight - window.innerHeight;
    const randomY = Math.floor(scrollpos * maxScrollY);
    window.scrollTo({top: randomY});

    function clip(x, min, max) {
        return Math.max(min, Math.min(max, x));
    }

    // blank all nodes that are not fully contained in viewport
    function blank(node) {
        if (node.nodeType !== Node.ELEMENT_NODE) {
            return false;
        }
        const bbox = getBoundingBox(node);
        if (!bbox) {
            return false;
        }
        const visible = (
            bbox[0] >= 0 &&
            bbox[1] >= 0 &&
            bbox[2] < viewportWidth &&
            bbox[3] < viewportHeight
        );
        if (visible) {
            return true;  // keep parents visible
        }
        // Because the DOM is a tree, we cannot just hide all nodes that are out of
        // the viewport, but we need to keep parents of visible nodes also visible.
        // However, nodes that contain text must be hidden if they are not fully
        // contained in the viewport.
        let keepVisible = false;
        for (const child of node.childNodes) {
            if (child.nodeType === Node.TEXT_NODE && child.textContent.trim()) {
                keepVisible = false;
                break;
            }
            else if (child.nodeType === Node.ELEMENT_NODE) {
                keepVisible |= blank(child);
            }
        }
        if (!keepVisible) {
            node.style.visibility = 'hidden';
        }
        return keepVisible;
    }
    blank(document.body);

    function isVisible(element) {
        if (element.nodeType === Node.TEXT_NODE) {
            element = element.parentNode; // Check visibility based on the parent node for text nodes
        }
        if (element.nodeType === Node.ELEMENT_NODE) {
            const style = window.getComputedStyle(element);
            const minHeight = ['TABLE', 'TBODY', 'TR'].includes(element.tagName) ? 10 : 1;
            // console.log(element.tagName, style.display, style.visibility, style.opacity, element.offsetWidth, element.offsetHeight);
            if (style.display === 'none' || style.visibility !== 'visible' || parseFloat(style.opacity) == 0) {
                return false;
            }
            else if (element.tagName === 'BR') {
                // BR has zero width and height
                return true;
            }
            else if (element.offsetWidth == 0 || element.offsetHeight < minHeight) {
                // Preserve whitespace-only elements (spaces, nbsp, etc.) even if they have zero width
                const textContent = element.textContent;
                const isWhitespaceOnly = textContent && /^\s*$/.test(textContent);
                if (isWhitespaceOnly) {
                    return true;
                }
                if (debug) console.log(`[ISVISIBLE] ${element.tagName} not visible due to zero width or height`);
                return false;
            }
            return true;
        }
        return false;
    }

    function getBoundingBox(node) {
        if (!isVisible(node)) {
            return null;
        }
        if (node.nodeType === Node.TEXT_NODE) {
            if (!node.textContent.trim()) {
                return null;
            }
            const range = document.createRange();
            range.selectNode(node);
            const rect = range.getBoundingClientRect();
            if (rect.width == 0 && rect.height == 0) {
                if (debug) console.log(`[BBOX] Zero-size text node rejected`);
                return null;
            }
            return [
                Math.floor(rect.left),
                Math.floor(rect.top),
                Math.ceil(rect.right),
                Math.ceil(rect.bottom)
            ];
        } else if (node.nodeType === Node.ELEMENT_NODE) {
            const rect = node.getBoundingClientRect();
            if (rect.width == 0 && rect.height == 0) {
                if (debug) console.log(`[BBOX] Zero-size element rejected: ${node.tagName}`);
                return null;
            }
            // use bounding client rect for tables and nodes with border
            if (node.tagName == 'TBODY' || node.clientLeft > 0 || node.clientTop > 0) {
                return [
                    Math.floor(rect.left),
                    Math.floor(rect.top),
                    Math.ceil(rect.right),
                    Math.ceil(rect.bottom)
                ];
            }
            // take union of text nodes' bounding boxes to exclude empty space
            let bbox = null;
            for (const child of node.childNodes) {
                const childBox = getBoundingBox(child);
                if (!childBox) continue;
                if (!bbox) {
                    bbox = [...childBox];
                } else {
                    bbox[0] = Math.min(bbox[0], childBox[0]);
                    bbox[1] = Math.min(bbox[1], childBox[1]);
                    bbox[2] = Math.max(bbox[2], childBox[2]);
                    bbox[3] = Math.max(bbox[3], childBox[3]);
                }
            }
            // extend left edge to include list marker (bullet etc.)
            if (bbox && node.tagName === 'LI') {
                // Get the marker's width using computed style
                let markerWidth = parseFloat(window.getComputedStyle(node, '::marker').width) || 0;
                const markerContent = window.getComputedStyle(node, '::marker').getPropertyValue('content');
                if (['normal', 'disc', 'circle', 'square'].includes(markerContent)) {
                    markerWidth += 15;
                }
                else if (markerContent === "decimal") {
                    markerWidth += 10;
                }
                bbox[0] -= clip(markerWidth, 0, bbox[0]);
            }
            
            return bbox;
        }
        return null;
    }

    function getMarkerText(listItem) {
        const markerContent = window.getComputedStyle(listItem, '::marker')
            .getPropertyValue('content');
        
        // If content is 'none', return empty string
        if (markerContent === 'none') {
            return '';
        }
        
        // If content is not 'normal', remove the quotes and return
        if (markerContent !== 'normal') {
            return markerContent.replace(/^["'](.*)["']$/, '$1');
        }
        
        // For 'normal' content, determine the marker based on list style type
        const listStyleType = window.getComputedStyle(listItem)
            .getPropertyValue('list-style-type');
            
        // For unordered lists with standard markers
        if (listStyleType === 'disc') return '•';
        if (listStyleType === 'circle') return '○';
        if (listStyleType === 'square') return '■';
        
        // For ordered lists
        if (listStyleType === 'decimal' || listStyleType === 'none') {
            const ol = listItem.closest('ol');
            if (!ol) return '';
            
            const start = parseInt(ol.getAttribute('start') || '1');
            const items = Array.from(ol.children);
            const index = items.indexOf(listItem);
            
            return (start + index) + '.';
        }
        
        // For alphabetical lists
        if (listStyleType === 'lower-alpha' || listStyleType === 'lower-latin') {
            const ol = listItem.closest('ol');
            if (!ol) return '';
            
            const start = parseInt(ol.getAttribute('start') || '1');
            const items = Array.from(ol.children);
            const index = items.indexOf(listItem);
            const charCode = 96 + (start + index); // 'a' is 97
            
            return String.fromCharCode(charCode) + '.';
        }
        
        if (listStyleType === 'upper-alpha' || listStyleType === 'upper-latin') {
            const ol = listItem.closest('ol');
            if (!ol) return '';
            
            const start = parseInt(ol.getAttribute('start') || '1');
            const items = Array.from(ol.children);
            const index = items.indexOf(listItem);
            const charCode = 64 + (start + index); // 'A' is 65
            
            return String.fromCharCode(charCode) + '.';
        }
        
        // Default to empty if we can't determine
        return '';
    }

    // wikipedia uses li::after for separators
    function getAfterContent(listItem) {
        const afterContent = window.getComputedStyle(listItem, '::after').getPropertyValue('content');
        
        // If content is 'none', return empty string
        if (afterContent === 'none') {
            return '';
        }
        
        // Remove the quotes around the content value
        return afterContent.replace(/^["'](.*)["']$/, '$1');
    }

    function cloneVisible(node) {
        if (node.nodeType === Node.TEXT_NODE) {
            return node.cloneNode(false);
        }
        if (node.nodeType === Node.ELEMENT_NODE) {
            if (['STYLE', 'SCRIPT', 'LINK', 'CANVAS', 'SVG'].includes(node.tagName)) {
                return null;
            }
            const clone = node.cloneNode(false);
            for (const attr of [...clone.attributes]) {
                if (['class', 'id', 'rel', 'role', 'about'].includes(attr.name) || attr.name.startsWith('data-')) {
                    clone.removeAttribute(attr.name);
                }
            }
            const afterContent = getAfterContent(node);
            if (afterContent) {
                clone.setAttribute('data-after', afterContent);
            }
            for (const child of node.childNodes) {
                if (isVisible(child)) {
                    const visibleChild = cloneVisible(child);
                    if (visibleChild) {
                        clone.appendChild(visibleChild);
                    }
                }
            }
            return clone;
        }
        return null;
    }

    function getContent(node) {
        if (node.nodeType === Node.TEXT_NODE) {
            return node.nodeValue.trim();
        }
        if (!isVisible(node)) {
            return '';
        }
        let visibleNode = cloneVisible(node);
        // console.log("original\n", node.outerHTML, "\n\n");
        // console.log("visible\n", visibleNode.outerHTML, "\n\n");
        if (visibleNode.tagName === 'LI') {
            return '<li>' + getMarkerText(node) + ' ' + visibleNode.innerHTML + '</li>';
        }
        return visibleNode.outerHTML;
    }

    function getCategory(node) {
        const tagName = node.tagName;
        
        if (['P', 'A', 'CENTER', 'CITE', 'DIV', 'DT', 'DD', 'I', 'B', 'U', 'S', 'EM', 'STRONG', 'TT', 'ABBR', 'SAMP', 'SMALL', 'BIG', 'MARK', 'SPAN', 'PRE', 'CODE', 'SUB', 'SUP', 'BLOCKQUOTE'].includes(tagName)) {
            return "Text";
        } else if (tagName === 'H1') {
            return "Title";
        } else if (['H2', 'H3', 'H4', 'H5', 'H6'].includes(tagName)) {
            return "Section-header";
        } else if (tagName == 'LI') {
            return "List-item";
        } else if (tagName === 'IMG') {
            return "Picture";
        } else if (tagName === 'TBODY') {
            return "Table";
        } else if (['CAPTION', 'FIGCAPTION'].includes(tagName)) {
            return "Caption";
        } else if (tagName === 'HEADER') {
            return "Page-header";
        } else if (tagName === 'FOOTER') {
            return "Page-footer";
        } else {
            throw new Error(`Failed to choose category for HTML element: ${tagName}: ${node.outerHTML.substr(0, 100)}...`);
        }
    }

    function getNodeResult(node) {
        const bbox = getBoundingBox(node);
        if (!bbox) {
            return [];
        }

        // clip table boxes (tables can be partially visible)
        if (node.tagName == 'TBODY') {
            bbox[0] = clip(bbox[0], 0, viewportWidth - 1);
            bbox[1] = clip(bbox[1], 0, viewportHeight - 1);
            bbox[2] = clip(bbox[2], 0, viewportWidth - 1);
            bbox[3] = clip(bbox[3], 0, viewportHeight - 1);
        }

        if (bbox[0] < 0 || bbox[0] >= viewportWidth ||
            bbox[1] < 0 || bbox[1] >= viewportHeight ||
            bbox[2] < 0 || bbox[2] >= viewportWidth ||
            bbox[3] < 0 || bbox[3] >= viewportHeight ||
            bbox[0] >= bbox[2] || bbox[1] >= bbox[3]) {
            throw new Error(`Invalid bounding box: ${bbox}: (${viewportWidth}x${viewportHeight}) ${node.tagName}: ${node.outerHTML.substr(0, 100)}...`);
        }

        return [{
            bbox: bbox,
            category_id: getCategory(node),
            content: getContent(node),
        }];
    }

    const extractTags = ['P', 'IMG', 'TBODY', 'CAPTION', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'LI', 'DD', 'DT', 'FOOTER', 'HEADER'];
    function traverse(node) {
        if (!isVisible(node)) {
            return [];
        }

        if (node.nodeType === Node.ELEMENT_NODE) {
            const tagName = node.tagName.toUpperCase();

            if (extractTags.includes(tagName)) {
                return getNodeResult(node);
            }

            // Check if node contains direct text nodes
            for (const childNode of node.childNodes) {
                if (childNode.nodeType === Node.TEXT_NODE && childNode.textContent.trim()) {
                    if (debug) console.log(`Processing element with text: ${tagName}`);
                    return getNodeResult(node);
                }
            }

            // Recursively process child nodes and combine results
            if (debug) console.log(`Recursing children of: ${tagName}`);
            let nodeResults = [];
            for (const childNode of node.childNodes) {
                nodeResults = nodeResults.concat(traverse(childNode));
            }
            return nodeResults;
        }

        return [];
    }

    return traverse(document.body);
}
