// Mappa dei campi voucher con i loro nomi descrittivi
const voucherFieldsMap = {
  "voucher": "descrivimi il voucher aziendale?",
  "voucher-code": "descrivimi il voucher-code aziendale?",
  "voucher-type": "descrivimi il voucher-type aziendale?", 
  "voucher-value": "descrivimi il voucher-value aziendale?",
  "voucher-expiry": "descrivimi il voucher-expiry aziendale?",
  "voucher-recipient": "descrivimi il voucher-recipient aziendale?"
};

// Funzione per inviare messaggio automatico al chatbot
function sendFieldMessage(fieldDescription) {
    if (chatWindow.classList.contains('visible')) {
        // Invia direttamente al server senza modificar userInput
        sendMessageSilent(fieldDescription);
    }
}

// Funzione simile a sendMessage, ma non aggiunge il messaggio al chatBox
async function sendMessageSilent(text) {
    if (!text || text.trim() === "") return;

    // Mostra solo indicatore di typing
    showTypingIndicator();

    try {
        const response = await fetch('/echo', {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: new URLSearchParams({text_input: text})
        });

        if (response.ok) {
            const result = await response.text();
            hideTypingIndicator();
            // Aggiungi SOLO la risposta del chatbot
            addMessage(result, 'ai');
        } else {
            hideTypingIndicator();
            addMessage("Errore: impossibile elaborare la richiesta", 'ai');
        }
    } catch (error) {
        hideTypingIndicator();
        addMessage("Errore: impossibile connettersi al server", 'ai');
        console.error(error);
    }
}

// Aggiungi listener a tutti i campi voucher
Object.keys(voucherFieldsMap).forEach(fieldId => {
    const field = document.getElementById(fieldId);

    // Focus o tab
    field.addEventListener('focus', () => sendFieldMessage(voucherFieldsMap[fieldId]));

    // Se è select, anche onChange
    if (field.tagName.toLowerCase() === 'select') {
        field.addEventListener('change', () => sendFieldMessage(voucherFieldsMap[fieldId]));
    }
});

// Resto del codice voucher.js (validazione, ecc.)
document.getElementById("validate-btn").addEventListener("click", () => {
  const code = document.getElementById("voucher-code").value.trim().toUpperCase();
  const type = document.getElementById("voucher-type").value;
  const value = document.getElementById("voucher-value").value.trim();
  const expiry = document.getElementById("voucher-expiry").value;
  const recipient = document.getElementById("voucher-recipient").value.trim();

  const resultDiv = document.getElementById("validation-result");
  const detailsDiv = document.getElementById("voucher-details");

  resultDiv.textContent = "";
  detailsDiv.classList.add("hidden");

  // Validazione base
  if (!code || code.length !== 12 || !/^[A-Z0-9]+$/.test(code)) {
    resultDiv.textContent = "❌ Codice non valido. Deve avere 12 caratteri alfanumerici.";
    resultDiv.className = "validation-result invalid";
    return;
  }

  if (!type || !value || !expiry || !recipient) {
    resultDiv.textContent = "❌ Compila tutti i campi obbligatori.";
    resultDiv.className = "validation-result invalid";
    return;
  }

  // Mostra dettagli
  resultDiv.textContent = "✅ Voucher pronto!";
  resultDiv.className = "validation-result valid";

  document.getElementById("detail-code").textContent = code;
  document.getElementById("detail-type").textContent = type;
  document.getElementById("detail-value").textContent = value;
  document.getElementById("detail-expiry").textContent = expiry;
  document.getElementById("detail-recipient").textContent = recipient;

  detailsDiv.classList.remove("hidden");
});
