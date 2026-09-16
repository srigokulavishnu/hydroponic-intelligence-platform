import React, { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Sparkles } from "lucide-react";
import { sendChatMessage } from "../services/api";
import { KnowledgeContext } from "./KnowledgeContext";

export const AssistantPanel = ({ plantId, plantState, messages, setMessages }) => {
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const examplePrompts = plantId ? [
    "What is the current nutrient level for this plant?",
    "Are the current temperature and pH optimal for this growth stage?",
    "Why does this plant look weak?",
    "What changed recently in the sensor history?"
  ] : [
    "What is the ideal pH range for Palak?",
    "What are the main growth stages of Palak?",
    "What does yellowing indicate in Palak leaves?",
    "How should EC be interpreted for Palak?"
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = async (text) => {
    if (!text.trim()) return;
    
    const userMsg = { role: "user", content: text };
    setMessages(prev => [...prev, userMsg]);
    setInputValue("");
    setLoading(true);

    try {
      const currentStateForChat = plantState ? {
        sensors: plantState.current_observation?.sensors,
        predictions: plantState.predictions
      } : null;
      
      const res = await sendChatMessage(plantId, text, currentStateForChat);
      const assistantMsg = { 
        role: "assistant", 
        content: res.response || "No response generated.",
        notes: res.context_notes,
        rawContext: res.raw_context,
        isError: res.status === "ERROR"
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      setMessages(prev => [...prev, { 
        role: "assistant", 
        content: "AI assistant is temporarily unavailable. " + err.message,
        isError: true 
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend(inputValue);
    }
  };

  const renderText = (text) => {
    return text.split("\n").map((line, i) => {
      if (!line.trim()) return <br key={i} />;
      const parts = line.split(/(\*\*.*?\*\*)/g);
      return (
        <p key={i} style={{ marginBottom: "0.5rem" }}>
          {parts.map((part, j) => {
            if (part.startsWith("**") && part.endsWith("**")) {
              return <strong key={j} style={{ color: "white" }}>{part.slice(2, -2)}</strong>;
            }
            return part;
          })}
        </p>
      );
    });
  };

  return (
    <div className="chat-container">
      <div style={{ padding: "1.25rem", borderBottom: "1px solid var(--border-subtle)", background: "rgba(0,0,0,0.2)" }}>
        <h3 style={{ display: "flex", alignItems: "center", gap: "0.5rem", margin: 0, fontSize: "1.1rem" }}>
          <Sparkles size={18} className="text-accent" /> Intelligence Engine
        </h3>
        <p className="text-xs text-secondary" style={{ marginTop: "0.25rem" }}>
          {plantId ? `Analyzing ${plantId}` : "Ask a general Palak question, or select a plant for plant-specific analysis."}
        </p>
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: "1.25rem", display: "flex", flexDirection: "column", gap: "1rem" }}>
        {messages.length === 0 ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: "1.5rem" }}>
            <div style={{ background: "rgba(16, 185, 129, 0.1)", padding: "1rem", borderRadius: "50%", boxShadow: "0 0 20px var(--accent-glow)" }}>
              <Bot size={36} className="text-accent" />
            </div>
            <div className="text-sm text-secondary text-center">
              {plantId ? `How can I assist with ${plantId}?` : "Ask about Palak, hydroponics, pH, EC, growth, or plant health."}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem", width: "100%", marginTop: "1rem" }}>
              {examplePrompts.map((prompt, i) => (
                <button 
                  key={i} 
                  className="btn" 
                  style={{ justifyContent: "flex-start", padding: "0.75rem 1rem", fontSize: "0.85rem", background: "rgba(0,0,0,0.3)" }}
                  onClick={() => handleSend(prompt)}
                >
                  "{prompt}"
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={msg.role === "user" ? "chat-bubble-user" : "chat-bubble-ai"} style={{ 
              alignSelf: msg.role === "user" ? "flex-end" : "flex-start",
              maxWidth: "85%",
              padding: "1rem 1.25rem",
              borderRadius: "var(--radius-md)",
              borderTopRightRadius: msg.role === "user" ? 0 : "var(--radius-md)",
              borderTopLeftRadius: msg.role === "assistant" ? 0 : "var(--radius-md)",
              boxShadow: "0 4px 12px rgba(0,0,0,0.1)"
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem", opacity: 0.8, color: msg.role === "user" ? "#000" : "var(--text-secondary)" }}>
                {msg.role === "user" ? <User size={14} /> : <Bot size={14} />}
                <span style={{ fontSize: "0.7rem", textTransform: "uppercase", fontWeight: "700", letterSpacing: "0.05em" }}>{msg.role}</span>
              </div>
              <div className="text-sm" style={{ wordBreak: "break-word", lineHeight: 1.6 }}>
                {msg.role === "assistant" ? renderText(msg.content) : msg.content}
              </div>
              
              {msg.notes && <KnowledgeContext notes={msg.notes} rawContext={msg.rawContext} />}
            </div>
          ))
        )}
        
        {loading && (
          <div className="chat-bubble-ai" style={{ alignSelf: "flex-start", maxWidth: "80%", padding: "1rem 1.25rem", borderRadius: "var(--radius-md)", borderTopLeftRadius: 0 }}>
            <div className="text-sm text-accent" style={{ display: "flex", alignItems: "center", gap: "0.75rem", fontWeight: "500" }}>
              <div style={{ animation: "pulse-glow 1.5s infinite", width: "8px", height: "8px", background: "var(--accent)", borderRadius: "50%" }}></div>
              {plantId ? "Processing environmental & visual context..." : "Querying knowledge base..."}
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div style={{ padding: "1.25rem", borderTop: "1px solid var(--border-subtle)", background: "rgba(0,0,0,0.2)" }}>
        <div style={{ position: "relative" }}>
          <input 
            type="text" 
            className="input" 
            placeholder={plantId ? `Ask about ${plantId}...` : "Ask about Palak, hydroponics, pH, EC, growth, or plant health..."}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            style={{ paddingRight: "3.5rem", borderRadius: "var(--radius-lg)", background: "rgba(0,0,0,0.4)" }}
          />
          <button 
            onClick={() => handleSend(inputValue)}
            disabled={loading || !inputValue.trim()}
            style={{ 
              position: "absolute", right: "0.35rem", top: "50%", transform: "translateY(-50%)", 
              width: "2.5rem", height: "2.5rem", borderRadius: "50%",
              background: inputValue.trim() ? "var(--accent)" : "rgba(255,255,255,0.1)",
              color: inputValue.trim() ? "#000" : "var(--text-muted)",
              border: "none", display: "flex", alignItems: "center", justifyContent: "center",
              cursor: inputValue.trim() && !loading ? "pointer" : "not-allowed",
              transition: "all 0.2s"
            }}
          >
            <Send size={16} style={{ marginLeft: inputValue.trim() ? "2px" : "0" }} />
          </button>
        </div>
      </div>
    </div>
  );
};
