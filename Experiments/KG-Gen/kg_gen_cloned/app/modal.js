class Modal {
    constructor(id, title, content, options = {}) {
        this.id = id;
        this.title = title;
        this.content = content;
        this.options = {
            width: '500px',
            height: 'auto',
            closable: true,
            backdrop: true,
            centered: true,
            selector: null,
            ...options
        };
        
        this.modalContainer = null;
        this.dialog = null;
        this.cleanup = null;
        this.isOpen = false;
        
        // If selector is provided, get content from HTML
        if (this.options.selector) {
            this.loadContentFromSelector();
        }
        
        this.init();
    }

    loadContentFromSelector() {
        const element = document.querySelector(this.options.selector);
        if (!element) {
            console.error(`Modal: Element with selector "${this.options.selector}" not found`);
            return;
        }

        // Extract title from element's data attributes if not provided
        if (!this.title || this.title === 'Modal') {
            this.title = element.dataset.title || element.dataset.modalTitle || 'Modal';
        }
        
        // Extract width from data attributes if not provided in options
        if (!this.options.width || this.options.width === '500px') {
            this.options.width = element.dataset.width || '500px';
        }

        // Handle template elements
        if (element.tagName === 'TEMPLATE') {
            // Clone the template content - element.content is a DocumentFragment
            const fragment = element.content.cloneNode(true);
            // Create a container div to hold the fragment content
            const container = document.createElement('div');
            container.appendChild(fragment);
            this.content = container;
        } else {
            // Clone regular element content
            this.content = element.cloneNode(true);
        }
        
        // Remove the original element from DOM if data-remove is true
        if (element.dataset.remove === 'true') {
            element.remove();
        }
    }

    // Static method to create modal from HTML selector
    static fromHTML(selector, options = {}) {
        const element = document.querySelector(selector);
        if (!element) {
            console.error(`Modal: Element with selector "${selector}" not found`);
            return null;
        }

        // Extract title from element's data-title attribute or use a default
        const title = element.dataset.title || element.dataset.modalTitle || 'Modal';
        
        // Extract width from data attributes
        const width = element.dataset.width || options.width || '500px';
        
        // Clone the element content to avoid removing it from DOM
        const content = element.cloneNode(true);
        
        // Remove the original element from DOM if data-remove is true
        if (element.dataset.remove === 'true') {
            element.remove();
        }

        // Create modal with cloned content
        const modal = new Modal(
            element.dataset.modalId || `modal-${Date.now()}`,
            title,
            content,
            {
                ...options,
                width
            }
        );

        return modal;
    }

    // Static method to create modal from template
    static fromTemplate(templateId, options = {}) {
        return new Modal(
            options.id || `modal-${Date.now()}`,
            options.title || 'Modal',
            null,
            {
                ...options,
                selector: `#${templateId}`
            }
        );
    }

    init() {
        this.createModal();
        this.setupEvents();
        this.show();
    }

    createModal() {
        // Create modal container
        this.modalContainer = document.createElement('div');
        this.modalContainer.id = `modal-container-${this.id}`;
        this.modalContainer.className = this.options.backdrop === false ? 'modal-container no-backdrop' : 'modal-container';
        this.modalContainer.setAttribute('aria-hidden', 'true');
        document.body.appendChild(this.modalContainer);

        // Create dialog
        this.dialog = document.createElement('div');
        this.dialog.id = this.id;
        this.dialog.className = 'modal-dialog';
        this.dialog.setAttribute('role', 'dialog');
        this.dialog.setAttribute('aria-labelledby', `${this.id}-title`);
        this.dialog.setAttribute('aria-modal', 'true');
        
        // Set dimensions
        this.dialog.style.width = this.options.width;
        if (this.options.height !== 'auto') {
            this.dialog.style.height = this.options.height;
        }

        // Create dialog HTML
        this.dialog.innerHTML = `
            <div class="modal-header">
                <h2 id="${this.id}-title" class="modal-title">${this.title}</h2>
                ${this.options.closable ? `
                    <button class="modal-close" aria-label="Close dialog" type="button">
                      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24">
                            <path fill="currentColor" d="M5.293 5.293a1 1 0 0 1 1.414 0L12 10.586l5.293-5.293a1 1 0 1 1 1.414 1.414L13.414 12l5.293 5.293a1 1 0 0 1-1.414 1.414L12 13.414l-5.293 5.293a1 1 0 0 1-1.414-1.414L10.586 12L5.293 6.707a1 1 0 0 1 0-1.414z"/>
                        </svg>
                    </button>
                ` : ''}
            </div>
            <div class="modal-body">
                <div class="modal-content"></div>
            </div>
        `;

        // Insert content
        const contentContainer = this.dialog.querySelector('.modal-content');
        this.insertContent(contentContainer);

        // Add to container
        this.modalContainer.appendChild(this.dialog);
    }

    insertContent(container) {
        if (typeof this.content === 'string') {
            container.innerHTML = this.content;
        } else if (this.content instanceof Element) {
            container.appendChild(this.content);
        } else if (typeof this.content === 'function') {
            const rendered = this.content();
            if (typeof rendered === 'string') {
                container.innerHTML = rendered;
            } else if (rendered instanceof Element) {
                container.appendChild(rendered);
            }
        }
        
        // Call external function to re-attach event handlers if it exists
        if (window.reattachModalEventHandlers && typeof window.reattachModalEventHandlers === 'function') {
            window.reattachModalEventHandlers(container);
        }
        
        // Re-setup tab trap after content is inserted and event handlers are re-attached
        if (this.isOpen) {
            // Use setTimeout to ensure DOM is fully updated
            setTimeout(() => {
                this.setupTabTrap();
            }, 0);
        }
    }

    setupEvents() {
        // Close button
        if (this.options.closable) {
            const closeBtn = this.dialog.querySelector('.modal-close');
            if (closeBtn) {
                closeBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    this.close();
                });
            }
        }

        // Escape key
        this.escapeHandler = (e) => {
            if (e.key === 'Escape' && this.isOpen) {
                this.close();
            }
        };
        document.addEventListener('keydown', this.escapeHandler);

        // Backdrop click
        if (this.options.backdrop) {
            this.backdropHandler = (e) => {
                if (e.target === this.modalContainer) {
                    this.close();
                }
            };
            this.modalContainer.addEventListener('click', this.backdropHandler);
        }

        // Store cleanup function
        this.cleanup = () => {
            document.removeEventListener('keydown', this.escapeHandler);
            if (this.options.backdrop) {
                this.modalContainer.removeEventListener('click', this.backdropHandler);
            }
            // Clean up tab trap
            if (this.tabTrapHandler) {
                document.removeEventListener('keydown', this.tabTrapHandler);
            }
        };
    }

    getFocusableElements() {
        // Get all focusable elements within the modal dialog
        const focusableSelectors = [
            'button:not([disabled])',
            '[href]',
            'input:not([disabled])',
            'select:not([disabled])',
            'textarea:not([disabled])',
            '[tabindex]:not([tabindex="-1"]):not([disabled])'
        ].join(', ');
        
        return Array.from(this.dialog.querySelectorAll(focusableSelectors))
            .filter(element => {
                // Additional checks for visibility
                const style = window.getComputedStyle(element);
                return style.display !== 'none' && 
                       style.visibility !== 'hidden' && 
                       !element.hasAttribute('aria-hidden');
            });
    }

    setupTabTrap() {
        // Remove existing tab trap handler if it exists
        if (this.tabTrapHandler) {
            document.removeEventListener('keydown', this.tabTrapHandler);
        }

        this.tabTrapHandler = (e) => {
            if (e.key === 'Tab' && this.isOpen) {
                const focusableElements = this.getFocusableElements();
                
                if (focusableElements.length === 0) {
                    // No focusable elements, prevent default tab behavior
                    e.preventDefault();
                    return;
                }

                const firstElement = focusableElements[0];
                const lastElement = focusableElements[focusableElements.length - 1];

                if (e.shiftKey) {
                    // Shift + Tab (backwards)
                    if (document.activeElement === firstElement) {
                        // Focus last element
                        e.preventDefault();
                        lastElement.focus();
                    }
                } else {
                    // Tab (forwards)
                    if (document.activeElement === lastElement) {
                        // Focus first element
                        e.preventDefault();
                        firstElement.focus();
                    }
                }
            }
        };

        document.addEventListener('keydown', this.tabTrapHandler);
    }

    show() {
        if (this.isOpen) return;
        
        this.modalContainer.hidden = false;
        this.modalContainer.setAttribute('aria-hidden', 'false');
        this.isOpen = true;

        // Setup tab trapping
        this.setupTabTrap();

        // Focus management
        const focusableElements = this.getFocusableElements();
        if (focusableElements.length > 0) {
            focusableElements[0].focus();
        } else {
            this.dialog.focus();
        }

        // Add to global registry
        if (!window.activeModals) {
            window.activeModals = new Map();
        }
        window.activeModals.set(this.id, this);
    }

    close() {
        if (!this.isOpen) return;

        // Cleanup events
        if (this.cleanup) {
            this.cleanup();
        }

        // Clean up tab trap handler
        if (this.tabTrapHandler) {
            document.removeEventListener('keydown', this.tabTrapHandler);
            this.tabTrapHandler = null;
        }

        // Hide modal
        this.modalContainer.hidden = true;
        this.modalContainer.setAttribute('aria-hidden', 'true');
        this.isOpen = false;

        // Remove from DOM
        this.modalContainer.remove();

        // Remove from registry
        if (window.activeModals) {
            window.activeModals.delete(this.id);
        }
    }

    updateContent(newContent) {
        this.content = newContent;
        const contentContainer = this.dialog.querySelector('.modal-content');
        if (contentContainer) {
            contentContainer.innerHTML = '';
            this.insertContent(contentContainer);
            
            // Re-setup tab trap after content update
            if (this.isOpen) {
                this.setupTabTrap();
            }
        }
    }

    // Static methods for global management
    static close(id) {
        if (window.activeModals && window.activeModals.has(id)) {
            window.activeModals.get(id).close();
        }
    }

    static closeAll() {
        if (window.activeModals) {
            for (const modal of window.activeModals.values()) {
                modal.close();
            }
        }
    }

    static isOpen(id) {
        return window.activeModals && window.activeModals.has(id);
    }

    static getActiveModals() {
        return window.activeModals ? Array.from(window.activeModals.keys()) : [];
    }
}

// Global API for backward compatibility
window.Modal = Modal;
window.openDialog = function(id, title, content, options) {
    return new Modal(id, title, content, options);
};
window.closeDialog = Modal.close;
window.closeAllDialogs = Modal.closeAll;
window.isDialogOpen = Modal.isOpen;
window.getActiveDialogs = Modal.getActiveModals;
