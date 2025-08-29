// Elementi DOM
const chatIcon = document.getElementById("chat-icon");
const chatWindow = document.getElementById("chat-window");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");
const chatBox = document.querySelector(".chat-box");
const typingIndicator = document.querySelector(".typing-indicator");

// Event listeners
chatIcon.addEventListener('click', toggleChat);
document.querySelector('.close-btn').addEventListener('click', toggleChat);
sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', function(e) {
    if(e.key === 'Enter') {
        sendMessage();
    }
});

// Funzioni principali
function toggleChat() {
    chatWindow.classList.toggle("visible");
    // Rimuovi il badge di notifica quando si apre la chat
    if (chatWindow.classList.contains("visible")) {
        document.querySelector('.notification-badge').style.display = 'none';
        userInput.focus(); // Focus automatico sull'input
    }
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (text === "") return;
    
    // Aggiungi messaggio utente
    addMessage(text, 'user');
    userInput.value = "";
    
    // Mostra indicatore typing
    showTypingIndicator();
    
    try {
        // Invia richiesta a FastAPI (stessa logica del progetto echo)
        const response = await fetch('/echo', {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: new URLSearchParams({text_input: text})
        });
        
        if (response.ok) {
            const result = await response.text();
            hideTypingIndicator();
            addMessage(result, 'ai');
        } else {
            hideTypingIndicator();
            addMessage("Errore: impossibile elaborare la richiesta", 'ai');
        }
    } catch (error) {
        hideTypingIndicator();
        addMessage("Errore: impossibile connettersi al server", 'ai');
        console.error('Errore:', error);
    }
}

function addMessage(text, sender) {
    const messageElement = document.createElement("div");
    messageElement.className = `message ${sender}`;
    
    // Aggiungi orario corrente
    const now = new Date();
    const time = now.getHours() + ':' + String(now.getMinutes()).padStart(2, '0');
    messageElement.innerHTML = `${text}<span class="message-time">${time}</span>`;
    
    chatBox.appendChild(messageElement);
    // Scroll to bottom con animazione smooth
    chatBox.scrollTop = chatBox.scrollHeight;
}

function showTypingIndicator() {
    typingIndicator.style.display = 'block';
    chatBox.appendChild(typingIndicator);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function hideTypingIndicator() {
    typingIndicator.style.display = 'none';
}

// Funzioni aggiuntive per migliorare UX
document.addEventListener('DOMContentLoaded', function() {
    // Messaggio di benvenuto animato dopo 1 secondo
    setTimeout(() => {
        if (!chatWindow.classList.contains('visible')) {
            // Pulsa l'icona per attirare l'attenzione
            chatIcon.style.animation = 'pulse 1.5s ease-in-out 3';
        }
    }, 1000);
});

// Gestione click fuori dalla chat per chiuderla
document.addEventListener('click', function(event) {
    const isClickInsideChat = chatWindow.contains(event.target);
    const isClickOnIcon = chatIcon.contains(event.target);
    
    if (!isClickInsideChat && !isClickOnIcon && chatWindow.classList.contains('visible')) {
        // Chiudi solo se si clicca fuori e la chat è aperta
        // toggleChat(); // Decommentare se si vuole chiudere cliccando fuori
    }
});

// Previeni il comportamento di default del form
userInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
        e.preventDefault();
    }
});
