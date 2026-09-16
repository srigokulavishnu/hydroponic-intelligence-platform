import React, { useState } from "react";
import { BookOpen, ChevronDown, ChevronRight } from "lucide-react";

export const KnowledgeContext = ({ notes, rawContext }) => {
  const [expanded, setExpanded] = useState(false);
  const [showJson, setShowJson] = useState(false);

  if (!notes || notes.length === 0) return null;

  return (
    <div style={{ marginTop: "0.75rem", background: "var(--bg-surface-hover)", borderRadius: "var(--radius-sm)", overflow: "hidden" }}>
      <button 
        style={{ 
          width: "100%", display: "flex", alignItems: "center", gap: "0.5rem", 
          padding: "0.5rem 0.75rem", background: "transparent", border: "none", 
          cursor: "pointer", color: "var(--text-secondary)", fontSize: "0.75rem", fontWeight: "500" 
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <BookOpen size={14} />
        Why this answer? (Knowledge Context)
        <div style={{ marginLeft: "auto" }}>
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </div>
      </button>

      {expanded && (
        <div style={{ padding: "0 0.75rem 0.75rem", fontSize: "0.75rem", color: "var(--text-secondary)" }}>
          <ul style={{ paddingLeft: "1.25rem", margin: 0, marginBottom: "0.5rem" }}>
            {notes.map((note, idx) => (
              <li key={idx} style={{ marginBottom: "0.25rem" }}>{note}</li>
            ))}
          </ul>
          
          {rawContext && (
            <div style={{ marginTop: "0.5rem", borderTop: "1px dashed var(--border-subtle)", paddingTop: "0.5rem" }}>
              <button 
                onClick={() => setShowJson(!showJson)}
                style={{ background: "transparent", border: "none", color: "var(--accent)", cursor: "pointer", fontSize: "0.7rem", padding: 0 }}
              >
                {showJson ? "Hide Raw Context JSON" : "View Raw Context JSON (For Demo)"}
              </button>
              
              {showJson && (
                <pre style={{ 
                  background: "rgba(0,0,0,0.5)", 
                  padding: "0.75rem", 
                  borderRadius: "var(--radius-sm)", 
                  overflowX: "auto",
                  marginTop: "0.5rem",
                  fontSize: "0.7rem",
                  color: "var(--text-primary)"
                }}>
                  {JSON.stringify(rawContext, null, 2)}
                </pre>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
