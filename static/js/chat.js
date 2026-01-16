/**
 * AI Tutor Chat Interface
 *
 * Handles chat functionality including:
 * - Sending/receiving messages
 * - Markdown rendering with syntax highlighting
 * - Dark mode toggle
 * - Session management
 * - Error handling
 */

// ===== Configuration =====
// CONFIG is defined in base.html template

// ===== State Management =====
const state = {
    isLoading: false,
    theme: localStorage.getItem('theme') || 'light'
};

// ===== DOM Elements =====
const elements = {
    chat: null,
    input: null,
    sendBtn: null,
    sendSpinner: null,
    darkModeToggle: null
};

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    elements.chat = document.getElementById('chat');
    elements.input = document.getElementById('input');
    elements.sendBtn = document.getElementById('sendBtn');
    elements.sendSpinner = document.getElementById('sendSpinner');
    elements.darkModeToggle = document.getElementById('darkModeToggle');

    // Initialize marked.js options
    marked.setOptions({
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                try {
                    return hljs.highlight(code, { language: lang }).value;
                } catch (err) {
                    console.error('Syntax highlighting error:', err);
                }
            }
            return hljs.highlightAuto(code).value;
        },
        breaks: true,
        gfm: true
    });

    // Set up event listeners
    setupEventListeners();

    // Apply saved theme
    applyTheme(state.theme);

    // Show initial greeting
    addMessage('Tutor', CONFIG.initialGreeting);

    // Focus input
    elements.input.focus();
});

// ===== Event Listeners =====
function setupEventListeners() {
    // Send on Enter (Shift+Enter for new line)
    elements.input.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMsg();
        }
    });

    // Dark mode toggle
    elements.darkModeToggle.addEventListener('click', toggleTheme);
}

// ===== Message Handling =====

/**
 * Add a message to the chat interface
 * @param {string} role - Either 'Student' or 'Tutor'
 * @param {string} text - The message text (markdown supported)
 */
function addMessage(role, text) {
    const div = document.createElement('div');
    const isUser = role === 'Student';

    div.className = `msg ${isUser ? 'user-message' : 'assistant-message'}`;

    // Create role label
    const roleSpan = document.createElement('span');
    roleSpan.className = 'role';
    roleSpan.textContent = role;

    // Create content div
    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';

    if (isUser) {
        // User messages: plain text with line breaks
        contentDiv.textContent = text;
    } else {
        // Assistant messages: render markdown with syntax highlighting
        try {
            const rawHtml = marked.parse(text);
            const cleanHtml = DOMPurify.sanitize(rawHtml);
            contentDiv.innerHTML = cleanHtml;

            // Apply syntax highlighting to code blocks
            contentDiv.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        } catch (err) {
            console.error('Markdown rendering error:', err);
            contentDiv.textContent = text; // Fallback to plain text
        }
    }

    div.appendChild(roleSpan);
    div.appendChild(contentDiv);
    elements.chat.appendChild(div);

    // Scroll to bottom
    scrollToBottom();
}

/**
 * Display an error message in the chat
 * @param {string} message - Error message to display
 */
function showError(message) {
    const div = document.createElement('div');
    div.className = 'error-message';
    div.textContent = `⚠️ Error: ${message}`;
    elements.chat.appendChild(div);
    scrollToBottom();
}

/**
 * Scroll chat container to bottom (smooth)
 */
function scrollToBottom() {
    elements.chat.scrollTo({
        top: elements.chat.scrollHeight,
        behavior: 'smooth'
    });
}

// ===== API Communication =====

/**
 * Send a chat message to the server
 */
async function sendMsg() {
    const text = elements.input.value.trim();

    // Validate input
    if (!text) {
        return;
    }

    // Prevent multiple simultaneous requests
    if (state.isLoading) {
        return;
    }

    // Clear input and disable button
    elements.input.value = '';
    setLoading(true);

    // Add user message to chat
    addMessage('Student', text);

    try {
        // Send request to server
        const response = await fetch(CONFIG.chatEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: text })
        });

        // Handle response
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        // Add assistant response to chat
        if (data.reply) {
            addMessage('Tutor', data.reply);
        } else {
            throw new Error('Empty response from server');
        }

    } catch (error) {
        console.error('Chat error:', error);
        showError(error.message || 'Failed to send message. Please try again.');
    } finally {
        setLoading(false);
        elements.input.focus();
    }
}

/**
 * Reset the chat session
 */
async function resetChat() {
    // Confirm with user
    if (!confirm('Are you sure you want to reset the conversation?')) {
        return;
    }

    // Prevent action during loading
    if (state.isLoading) {
        return;
    }

    setLoading(true);

    try {
        const response = await fetch(CONFIG.resetEndpoint, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        // Clear chat UI
        elements.chat.innerHTML = '';

        // Show confirmation message
        addMessage('Tutor', 'Okay, I cleared our session context. What assignment are you working on?');

    } catch (error) {
        console.error('Reset error:', error);
        showError('Failed to reset chat. Please refresh the page.');
    } finally {
        setLoading(false);
        elements.input.focus();
    }
}

// ===== UI State Management =====

/**
 * Set loading state for the UI
 * @param {boolean} loading - Whether the UI should show loading state
 */
function setLoading(loading) {
    state.isLoading = loading;

    if (loading) {
        elements.sendBtn.disabled = true;
        elements.sendBtn.classList.add('loading');
        elements.input.disabled = true;
    } else {
        elements.sendBtn.disabled = false;
        elements.sendBtn.classList.remove('loading');
        elements.input.disabled = false;
    }
}

// ===== Theme Management =====

/**
 * Toggle between light and dark themes
 */
function toggleTheme() {
    const newTheme = state.theme === 'light' ? 'dark' : 'light';
    state.theme = newTheme;
    localStorage.setItem('theme', newTheme);
    applyTheme(newTheme);
}

/**
 * Apply the specified theme
 * @param {string} theme - Either 'light' or 'dark'
 */
function applyTheme(theme) {
    const root = document.documentElement;
    const icon = elements.darkModeToggle.querySelector('.icon');
    const lightHighlight = document.getElementById('highlight-light');
    const darkHighlight = document.getElementById('highlight-dark');

    if (theme === 'dark') {
        root.setAttribute('data-theme', 'dark');
        icon.textContent = '☀️';
        elements.darkModeToggle.title = 'Switch to light mode';

        // Switch highlight.js theme
        lightHighlight.disabled = true;
        darkHighlight.disabled = false;
    } else {
        root.setAttribute('data-theme', 'light');
        icon.textContent = '🌙';
        elements.darkModeToggle.title = 'Switch to dark mode';

        // Switch highlight.js theme
        lightHighlight.disabled = false;
        darkHighlight.disabled = true;
    }
}

// ===== Utility Functions =====

/**
 * Format timestamp for display
 * @param {Date} date - Date object to format
 * @returns {string} Formatted time string
 */
function formatTime(date) {
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
    });
}

/**
 * Escape HTML to prevent XSS (backup for DOMPurify)
 * @param {string} text - Text to escape
 * @returns {string} HTML-escaped text
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Export functions for potential testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        addMessage,
        sendMsg,
        resetChat,
        toggleTheme
    };
}
