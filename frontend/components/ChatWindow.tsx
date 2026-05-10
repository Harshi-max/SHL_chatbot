"use client";

import { FormEvent, useMemo, useState } from "react";

type Role = "user" | "assistant";

type Message = {
  role: Role;
  content: string;
};

type Recommendation = {
  name: string;
  url: string;
  test_type: string;
};

type ChatResponse = {
  reply: string;
  recommendations: Recommendation[];
  end_of_conversation: boolean;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Describe the role you are hiring for, and I will suggest SHL Individual Test Solutions."
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [sessionEnded, setSessionEnded] = useState(false);
  const [error, setError] = useState("");

  const userMessagesCount = useMemo(
    () => messages.filter((m) => m.role === "user").length,
    [messages]
  );

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!input.trim() || loading || sessionEnded) return;

    const nextUserMessage: Message = { role: "user", content: input.trim() };
    const nextMessages = [...messages, nextUserMessage];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);
    setError("");

    try {
      const payload = {
        messages: nextMessages
      };
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data: ChatResponse = await response.json();
      setMessages((prev) => [...prev, { role: "assistant", content: data.reply }]);
      setRecommendations(data.recommendations ?? []);
      setSessionEnded(Boolean(data.end_of_conversation));
    } catch (submitError) {
      const message = submitError instanceof Error ? submitError.message : "Failed to call backend.";
      setError(message);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I could not reach the backend right now. Please try again."
        }
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 900, margin: "40px auto", padding: 16 }}>
      <h1 style={{ marginTop: 0 }}>SHL Conversational Recommender</h1>
      <p style={{ opacity: 0.8 }}>
        Stateless chat UI. Each request sends full `messages` history to backend `/chat`.
      </p>

      <div
        style={{
          border: "1px solid #334155",
          borderRadius: 10,
          padding: 16,
          minHeight: 360,
          background: "#111827"
        }}
      >
        {messages.map((message, idx) => (
          <div
            key={`${message.role}-${idx}`}
            style={{
              marginBottom: 12,
              textAlign: message.role === "user" ? "right" : "left"
            }}
          >
            <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 4 }}>{message.role}</div>
            <div
              style={{
                display: "inline-block",
                padding: "10px 12px",
                borderRadius: 8,
                background: message.role === "user" ? "#1d4ed8" : "#1f2937",
                maxWidth: "80%"
              }}
            >
              {message.content}
            </div>
          </div>
        ))}
      </div>

      <form onSubmit={onSubmit} style={{ marginTop: 12, display: "flex", gap: 8 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Example: Hiring a Java developer with client-facing responsibilities"
          style={{
            flex: 1,
            padding: 10,
            borderRadius: 8,
            border: "1px solid #334155",
            background: "#0f172a",
            color: "#e5e7eb"
          }}
          disabled={loading || sessionEnded}
        />
        <button
          type="submit"
          disabled={loading || sessionEnded}
          style={{
            padding: "10px 14px",
            borderRadius: 8,
            border: "none",
            background: "#2563eb",
            color: "white",
            cursor: "pointer"
          }}
        >
          {loading ? "Sending..." : "Send"}
        </button>
      </form>

      <div style={{ marginTop: 10, opacity: 0.8, fontSize: 13 }}>
        User turns: {userMessagesCount} / 8
      </div>

      {error && (
        <div style={{ marginTop: 8, color: "#fca5a5", fontSize: 13 }}>
          Error: {error}
        </div>
      )}

      <div style={{ marginTop: 18 }}>
        <h2 style={{ marginBottom: 8 }}>Recommendations</h2>
        {recommendations.length === 0 ? (
          <p style={{ opacity: 0.8 }}>No recommendations yet.</p>
        ) : (
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {recommendations.map((rec) => (
              <li key={rec.url} style={{ marginBottom: 8 }}>
                <a href={rec.url} target="_blank" rel="noreferrer">
                  {rec.name}
                </a>{" "}
                <span style={{ opacity: 0.8 }}>({rec.test_type || "N/A"})</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
