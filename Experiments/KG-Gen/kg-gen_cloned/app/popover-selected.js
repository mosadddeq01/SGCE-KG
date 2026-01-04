document.addEventListener('DOMContentLoaded', () => {
    const mainPopover = createPopover({
        viewSelectionBtn: document.getElementById('viewSelectionBtn'),
        selectionPopover: document.getElementById('selectionPopover'),
        closePopover: document.getElementById('closePopover'),
        selectionList: document.getElementById('selectionList'),
    });

    window.setMainPopoverContent = mainPopover.updateSelection;
    
    const mobilePopover = createPopover({
        viewSelectionBtn: document.getElementById('viewSelectionBtnMobile'),
        selectionPopover: document.getElementById('selectionPopoverMobile'),
        closePopover: document.getElementById('closePopoverMobile'),
        selectionList: document.getElementById('selectionListMobile'),
    });

    window.setMobilePopoverContent = mobilePopover.updateSelection;
});

function createPopover(config) {
    const {
        viewSelectionBtn,
        selectionPopover,
        closePopover,
        selectionList,
    } = config;

    if (!viewSelectionBtn || !selectionPopover || !selectionList) return;

    let cleanup = null;
    let customRenderer = null;

    function setContent(content) {
        if (typeof content === 'string') {
            selectionList.innerHTML = content;
            return;
        }
        if (content instanceof Element) {
            selectionList.innerHTML = '';
            selectionList.appendChild(content);
            return;
        }
        if (Array.isArray(content)) {
            if (content.length === 0) {
                selectionList.innerHTML = 'No items selected';
                return;
            }
            selectionList.innerHTML = content.map(item => `
                <div class="selection-item">
                    <strong>${item.label || item.id || 'Item'}</strong>
                    ${item.type ? `<br><small>Type: ${item.type}</small>` : ''}
                    ${item.cluster ? `<br><small>Cluster: ${item.cluster}</small>` : ''}
                </div>
            `).join('');
            return;
        }
        if (typeof content === 'function') {
            setContent(content());
            return;
        }
        selectionList.innerHTML = 'No items selected';
    }

    function updateSelection(content = []) {
        if (customRenderer) {
            setContent(customRenderer(content));
        } else {
            setContent(content);
        }
    }

    function showPopover() {
        if (cleanup) cleanup();
        selectionPopover.hidden = false;
        cleanup = FloatingUIDOM.autoUpdate(
            viewSelectionBtn,
            selectionPopover,
            () => {
                FloatingUIDOM.computePosition(viewSelectionBtn, selectionPopover, {
                    placement: 'bottom-end',
                    middleware: [
                        FloatingUIDOM.offset(8),
                        FloatingUIDOM.flip(),
                        FloatingUIDOM.shift({ padding: 8 })
                    ]
                }).then(({ x, y }) => {
                    Object.assign(selectionPopover.style, { left: `${x}px`, top: `${y}px` });
                });
            }
        );
        selectionPopover.setAttribute('aria-hidden', 'false');
        viewSelectionBtn.setAttribute('aria-expanded', 'true');
    }

    function hidePopover() {
        selectionPopover.hidden = true;
        if (cleanup) {
            cleanup();
            cleanup = null;
        }
        selectionPopover.setAttribute('aria-hidden', 'true');
        viewSelectionBtn.setAttribute('aria-expanded', 'false');
        viewSelectionBtn.focus();
    }

    viewSelectionBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (selectionPopover.hidden) {
            showPopover();
        } else {
            hidePopover();
        }
    });

    if (closePopover) {
        closePopover.addEventListener('click', hidePopover);
    }

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !selectionPopover.hidden) hidePopover();
    });

    document.addEventListener('click', (e) => {
        if (!selectionPopover.hidden && !selectionPopover.contains(e.target) && !viewSelectionBtn.contains(e.target)) {
            hidePopover();
        }
    });

    window.updateSelectionPopover = updateSelection;
    window.setSelectionPopoverRenderer = function(renderer) { customRenderer = typeof renderer === 'function' ? renderer : null; };
    window.openSelectionPopover = function(content) { if (content !== undefined) updateSelection(content); showPopover(); };
    window.closeSelectionPopover = function() { hidePopover(); };

    return {
        updateSelection,
    }
}