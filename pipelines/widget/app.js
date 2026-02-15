const messagesEl = document.getElementById("messages");
const form = document.getElementById("input-form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");

const API_BASE = window.location.origin;
const history = [];

function renderMarkdown(text) {
  let html = marked.parse(text);
  // Wrap tables in scrollable container
  html = html.replace(/<table>/g, '<div class="table-wrapper"><table>').replace(/<\/table>/g, '</table></div>');
  return html;
}

function addMessage(role, content) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant") {
    bubble.innerHTML = renderMarkdown(content);
  } else {
    bubble.textContent = content;
  }
  div.appendChild(bubble);
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

function createTypingBubble() {
  const div = document.createElement("div");
  div.className = "message assistant typing";
  div.id = "typing";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  div.appendChild(bubble);
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

function setLoading(loading) {
  sendBtn.disabled = loading;
  input.disabled = loading;
}

function hideSuggestions() {
  const el = document.getElementById("suggestions");
  if (el) el.remove();
}

async function loadSuggestions() {
  try {
    const res = await fetch(`${API_BASE}/suggestions`);
    const data = await res.json();
    const container = document.getElementById("suggestions");
    if (!container || !data.documents || data.documents.length === 0) return;

    // Generate varied questions per document
    const templates = [
      (d) => `Que documentos hay disponibles?`,
      (d) => `Resumeme ${d.description}`,
      (d) => `Requisitos principales`,
      (d) => `Fechas y plazos importantes`,
    ];

    for (const tpl of templates) {
      const text = tpl(data.documents[0]);
      const chip = document.createElement("button");
      chip.className = "suggestion-chip";
      chip.textContent = text;
      chip.addEventListener("click", () => {
        hideSuggestions();
        sendMessage(chip.textContent);
      });
      container.appendChild(chip);
    }
  } catch (e) {
    // Silently fail - suggestions are optional
  }
}

async function sendMessage(text) {
  if (!text.trim()) return;

  hideSuggestions();
  addMessage("user", text);
  history.push({ role: "user", content: text });
  setLoading(true);

  const bubble = createTypingBubble();
  let fullResponse = "";

  try {
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, history: history.slice(0, -1) }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith("data:")) {
          // Strip exactly one SSE space after "data:", preserve content whitespace
          const raw = line.slice(5);
          const data = raw.startsWith(" ") ? raw.slice(1) : raw;
          if (data === "") continue;
          fullResponse += data;
          bubble.innerHTML = renderMarkdown(fullResponse);
          messagesEl.scrollTop = messagesEl.scrollHeight;
        }
        if (line.startsWith("event:") && line.includes("done")) {
          break;
        }
      }
    }

    // Remove typing indicator
    const typingEl = document.getElementById("typing");
    if (typingEl) {
      typingEl.classList.remove("typing");
      typingEl.removeAttribute("id");
    }

    history.push({ role: "assistant", content: fullResponse });
  } catch (err) {
    bubble.textContent = "Error connecting to server. Please try again.";
    const typingEl = document.getElementById("typing");
    if (typingEl) {
      typingEl.classList.remove("typing");
      typingEl.removeAttribute("id");
    }
  }

  setLoading(false);
  input.focus();
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value;
  input.value = "";
  sendMessage(text);
});

// Load suggestions on startup
loadSuggestions();
