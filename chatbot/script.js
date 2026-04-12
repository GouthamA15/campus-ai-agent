// Simple frontend logic for the Interactive Campus Info AI Agent.
// - Sends the user's question to the FastAPI backend /ask endpoint
// - Renders messages in a basic chat-style interface

const API_BASE_URL = "http://127.0.0.1:8000"; 

const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

/**
 * Append a message bubble to the chat window.
 *
 * @param {"user"|"assistant"} role - Author of the message.
 * @param {string} text - Message content.
 */
function addMessage(role, text) {
  const messageEl = document.createElement("div");
  messageEl.classList.add("message", role === "user" ? "message-user" : "message-assistant");

  const bubbleEl = document.createElement("div");
  bubbleEl.classList.add("bubble");
  bubbleEl.textContent = text;

  messageEl.appendChild(bubbleEl);
  chatWindow.appendChild(messageEl);

  // Scroll to the latest message
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

/**
 * Show a temporary loading state for the assistant.
 */
function showLoading() {
  sendBtn.disabled = true;
  sendBtn.textContent = "...";
}

/**
 * Clear the loading state.
 */
function hideLoading() {
  sendBtn.disabled = false;
  sendBtn.textContent = "Send";
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = userInput.value.trim();
  if (!question) return;

  // Show user's message in the chat
  addMessage("user", question);

  // Clear input and set loading state
  userInput.value = "";
  showLoading();

  try {
    const response = await fetch(`${API_BASE_URL}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      // If backend returns an error, display a generic message
      addMessage(
        "assistant",
        "Sorry, I ran into an issue while fetching the answer. Please try again."
      );
    } else {
      const data = await response.json();
      addMessage("assistant", data.answer || "(No answer returned from the server.)");
    }
  } catch (error) {
    console.error("Error calling backend /ask endpoint", error);
    addMessage(
      "assistant",
      "Sorry, I could not connect to the server. Please ensure the backend is running."
    );
  } finally {
    hideLoading();
    userInput.focus();
  }
});

// Optionally, send a welcome message from the assistant on load
window.addEventListener("DOMContentLoaded", () => {
  addMessage(
    "assistant",
    "Hi! I'm the Kakatiya University Campus Info AI Agent. Ask me about admissions, departments, or campus facilities."
  );
});
