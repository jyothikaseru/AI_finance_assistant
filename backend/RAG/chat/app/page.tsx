"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Source {
  chunk_uid: string;
  ticker: string;
  fiscal_year: number;
  report_type: string;
  chunk_type: string;
  section?: string;
  page_start: number;
  similarity: number;
  content: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hello! I am **FinRAG**, your enterprise financial intelligence assistant. Ask me anything about corporate Annual Reports and SEC 10-K filings across any company or fiscal year.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [openSourceIndex, setOpenSourceIndex] = useState<number | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSubmit = async (e?: React.FormEvent, customQuery?: string) => {
    if (e) e.preventDefault();
    const query = customQuery || input;
    if (!query.trim() || loading) return;

    const userMessage: Message = { role: "user", content: query };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: query }),
      });

      const data = await response.json();

      if (response.ok) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.answer,
            sources: data.sources,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `⚠️ Error: ${data.error || "Failed to retrieve response"}`,
          },
        ]);
      }
    } catch (err: unknown) {
      const errorMsg =
        err instanceof Error ? err.message : "Connection failed";
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `⚠️ Connection Error: ${errorMsg}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const samplePrompts = [
    "What was Apple's total net sales in 2024?",
    "Compare Microsoft and Apple 2024 operating margins.",
    "What are Infosys top AI and cybersecurity risk factors?",
    "Show Reliance consolidated revenue and EBITDA summary.",
    "How did TCS perform in fiscal year 2024?",
  ];

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans">
      {/* Sidebar: Navigation & Quick Queries */}
      <div className="w-80 border-r border-slate-800 bg-slate-900/50 p-6 flex flex-col justify-between hidden md:flex">
        <div>
          <div className="flex items-center space-x-3 mb-8">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center font-bold text-black text-xl shadow-lg shadow-emerald-500/20">
              📊
            </div>
            <div>
              <h1 className="font-bold text-lg text-white">FinRAG Engine</h1>
              <p className="text-xs text-slate-400">Universal Financial Intelligence</p>
            </div>
          </div>

          <div className="space-y-4">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Sample Financial Queries
            </p>
            <div className="space-y-2">
              {samplePrompts.map((sampleQuery, i) => (
                <button
                  key={i}
                  onClick={() => handleSubmit(undefined, sampleQuery)}
                  className="w-full text-left text-xs bg-slate-800/60 hover:bg-slate-800 border border-slate-700/50 hover:border-emerald-500/50 rounded-lg p-2.5 text-slate-300 hover:text-emerald-400 transition leading-relaxed"
                >
                  {sampleQuery}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="text-xs text-slate-500 border-t border-slate-800/80 pt-4 flex items-center justify-between">
          <span>Supabase pgvector + Groq</span>
          <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
        </div>
      </div>

      {/* Main Chat Interface */}
      <div className="flex-1 flex flex-col h-full bg-slate-950">
        {/* Chat Feed */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 max-w-4xl w-full mx-auto">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`flex flex-col ${
                msg.role === "user" ? "items-end" : "items-start"
              }`}
            >
              <div
                className={`max-w-2xl rounded-2xl px-5 py-4 text-sm leading-relaxed shadow-md ${
                  msg.role === "user"
                    ? "bg-emerald-600 text-white rounded-br-none"
                    : "bg-slate-900 border border-slate-800 text-slate-200 rounded-bl-none"
                }`}
              >
                <div className="prose prose-invert max-w-none text-sm leading-relaxed font-sans space-y-3 [&_table]:w-full [&_table]:border [&_table]:border-slate-700 [&_table]:my-4 [&_th]:bg-slate-800 [&_th]:p-2.5 [&_th]:border [&_th]:border-slate-700 [&_th]:text-emerald-400 [&_td]:p-2.5 [&_td]:border [&_td]:border-slate-800 [&_td]:bg-slate-900/50 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:list-decimal [&_ol]:pl-5">
                 <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.content}
                  </ReactMarkdown>
                </div>
                {/* Grounded Sources Accordion */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-4 pt-3 border-t border-slate-800">
                    <button
                      onClick={() =>
                        setOpenSourceIndex(openSourceIndex === index ? null : index)
                      }
                      className="text-xs font-medium text-emerald-400 hover:text-emerald-300 flex items-center space-x-1.5 transition"
                    >
                      <span>📑 {msg.sources.length} Grounded Citations</span>
                      <span>{openSourceIndex === index ? "▲" : "▼"}</span>
                    </button>

                    {openSourceIndex === index && (
                      <div className="mt-3 space-y-2">
                        {msg.sources.map((src, srcIdx) => (
                          <div
                            key={srcIdx}
                            className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-3 text-xs"
                          >
                            <div className="flex items-center justify-between font-mono text-emerald-400 mb-1">
                              <span>
                                {src.ticker} {src.fiscal_year} ({src.report_type})
                              </span>
                              <span className="text-slate-500">
                                Page {src.page_start} | Match:{" "}
                                {(src.similarity * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="text-slate-400 text-[11px] font-medium mb-1">
                              Section: {src.section || "General"}
                            </div>
                            <div className="text-slate-300 font-mono text-[11px] max-h-28 overflow-y-auto bg-slate-900/90 p-2 rounded border border-slate-800/50 whitespace-pre-wrap">
                              {src.content}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex items-center space-x-3 text-slate-400 text-sm">
              <div className="w-4 h-4 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
              <span>Searching reports & synthesizing financial metrics...</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Universal Input Bar */}
        <div className="p-4 border-t border-slate-800 bg-slate-900/30">
          <form
            onSubmit={handleSubmit}
            className="max-w-4xl mx-auto flex items-center space-x-3"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask any financial question (e.g. 'Compare Apple and Microsoft 2024 revenue')..."
              className="flex-1 bg-slate-900 border border-slate-700/80 rounded-xl px-4 py-3.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="bg-emerald-500 hover:bg-emerald-400 disabled:bg-slate-800 disabled:text-slate-600 text-slate-950 font-semibold px-6 py-3.5 rounded-xl text-sm transition shadow-lg shadow-emerald-500/20 disabled:shadow-none"
            >
              Analyze
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}