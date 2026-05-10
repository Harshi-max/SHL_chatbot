const chatContainer = document.getElementById("chatContainer");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const typingRow = document.getElementById("typingRow");
const exportBtn = document.getElementById("exportBtn");

const state = {
  messages: [
    {
      role: "assistant",
      content:
        "Share role, seniority, and skills. I will suggest SHL Individual Test Solutions."
    }
  ],
  endOfConversation: false
};

function renderBubble(role, content) {
  const row = document.createElement("div");
  row.className = "w-full";
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role === "user" ? "bubble-user" : "bubble-assistant"}`;
  bubble.innerHTML = window.marked.parse(content);
  row.appendChild(bubble);
  chatContainer.appendChild(row);
}

function renderRecommendations(recommendations, replyText) {
  const wrapper = document.createElement("div");
  wrapper.className = "mt-2 mb-3";

  const label = document.createElement("div");
  label.className = "text-xs text-slate-400 mb-2";
  label.textContent = "Recommendations for this response";
  wrapper.appendChild(label);

  if (!recommendations || recommendations.length === 0) {
    const empty = document.createElement("div");
    empty.className = "text-sm text-slate-400";
    empty.textContent = "No recommendations yet. The assistant is asking for clarification.";
    wrapper.appendChild(empty);
    chatContainer.appendChild(wrapper);
    return;
  }

  const grid = document.createElement("div");
  grid.className = "grid gap-3 md:grid-cols-2";

  const isComparison = /compare|comparison/i.test(replyText) && recommendations.length === 2;
  recommendations.forEach((rec) => {
    const card = document.createElement("div");
    card.className = "card";
    const safeDesc = isComparison
      ? "Included in side-by-side comparison from catalog context."
      : "Catalog-grounded recommendation based on your request.";
    card.innerHTML = `
      <div class="flex items-center justify-between gap-2">
        <h3 class="font-semibold">${rec.name}</h3>
        <span class="rounded-full bg-indigo-500/25 px-2 py-1 text-xs">${rec.test_type || "N/A"}</span>
      </div>
      <p class="mt-2 text-sm text-slate-300">${safeDesc}</p>
      <a class="mt-3 inline-block text-sm text-sky-300 hover:text-sky-200" target="_blank" rel="noreferrer" href="${rec.url}">
        Official SHL Link
      </a>
    `;
    grid.appendChild(card);
  });

  wrapper.appendChild(grid);
  chatContainer.appendChild(wrapper);
}

function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function setLoading(loading) {
  sendBtn.disabled = loading || state.endOfConversation;
  chatInput.disabled = loading || state.endOfConversation;
  typingRow.classList.toggle("hidden", !loading);
}

function exportConversation() {
  const content = state.messages.map((m) => `### ${m.role}\n${m.content}`).join("\n\n");
  const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "shl-conversation.md";
  a.click();
  URL.revokeObjectURL(url);
}

async function submitMessage(event) {
  event.preventDefault();
  const text = chatInput.value.trim();
  if (!text || state.endOfConversation) return;

  state.messages.push({ role: "user", content: text });
  renderBubble("user", text);
  chatInput.value = "";
  setLoading(true);
  scrollToBottom();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: state.messages })
    });

    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }

    const payload = await response.json();
    state.messages.push({ role: "assistant", content: payload.reply });
    state.endOfConversation = Boolean(payload.end_of_conversation);
    renderBubble("assistant", payload.reply);
    renderRecommendations(payload.recommendations, payload.reply);
    scrollToBottom();
  } catch (error) {
    const fallback = "I could not process that just now. Please try again.";
    state.messages.push({ role: "assistant", content: fallback });
    renderBubble("assistant", fallback);
    scrollToBottom();
  } finally {
    setLoading(false);
  }
}

chatForm.addEventListener("submit", submitMessage);
exportBtn.addEventListener("click", exportConversation);
renderBubble("assistant", state.messages[0].content);
