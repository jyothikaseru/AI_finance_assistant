import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";
import Groq from "groq-sdk";

// 1. Initialize Supabase and Groq Clients
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const groqApiKey = process.env.GROQ_API_KEY!;

const supabase = createClient(supabaseUrl, supabaseKey);
const groq = new Groq({ apiKey: groqApiKey });

// 2. TypeScript Interface for Financial Chunks
interface FinancialChunk {
  id: number;
  chunk_uid: string;
  ticker: string;
  fiscal_year: number;
  report_type: string;
  chunk_type: string;
  section?: string;
  page_start: number;
  page_end: number;
  content: string;
  source?: string;
  similarity: number;
}

// 3. Local BAAI/BGE Embedding Pipeline (Exact Coordinate Match with Supabase)
type EmbedderPipeline = (
  text: string,
  options?: { pooling: string; normalize: boolean }
) => Promise<{ data: number[] | Float32Array }>;

let embedder: EmbedderPipeline | null = null;

async function getBAAIEmbedding(text: string): Promise<number[]> {
  try {
    if (!embedder) {
      const { pipeline } = await import("@xenova/transformers");
      embedder = (await pipeline(
        "feature-extraction",
        "Xenova/bge-base-en-v1.5"
      )) as unknown as EmbedderPipeline;
    }

    const output = await embedder(text, { pooling: "mean", normalize: true });
    return Array.from(output.data);
  } catch (err: unknown) {
    console.error("Local embedding fallback to keyword search:", err);
    return new Array(768).fill(0);
  }
}

// 4. Main Route Handler (POST /api/chat)
export async function POST(req: NextRequest) {
  try {
    const { message, ticker, fiscalYear } = await req.json();

    if (!message) {
      return NextResponse.json({ error: "Message is required" }, { status: 400 });
    }

    // A. Generate Query Vector
    const queryEmbedding = await getBAAIEmbedding(message);

    // B. Run Hybrid Search in Supabase
    const { data: chunks, error: rpcError } = await supabase.rpc("match_financial_chunks", {
      query_text: message,
      query_embedding: queryEmbedding,
      match_count: 6,
      filter_ticker: ticker || null,
      filter_year: fiscalYear ? parseInt(fiscalYear) : null,
      filter_type: null,
    });

    if (rpcError) {
      console.error("Supabase RPC Error:", rpcError);
      return NextResponse.json({ error: "Database retrieval failed" }, { status: 500 });
    }

    const typedChunks: FinancialChunk[] = chunks || [];

    // C. Construct Context from Retrieved Text & Markdown Tables
    const contextText = typedChunks
      .map(
        (c, i) =>
          `[Source ${i + 1} | Company: ${c.ticker} | Year: ${c.fiscal_year} | Section: ${c.section || "General"} | Page: ${c.page_start}]\n${c.content}`
      )
      .join("\n\n---\n\n");

        // D. Conversational Financial Analyst System Prompt
    const systemPrompt = `You are FinRAG, an intelligent, friendly financial assistant specializing in corporate Annual Reports and 10-K filings.

Context from Financial Reports:
${contextText || "No matching reports found in database."}

Instructions:
1. Tone & Style: Respond naturally, conversationally, and clearly like a professional financial consultant.
2. Grounding: Base your answer strictly on the provided context. If figures or companies are not in the context, politely state: "I don't have that specific data in the indexed annual reports."
3. Formatting: Present key numbers clearly in bullet points or a clean visual table, accompanied by a brief, easy-to-read explanation.
4. Citations: Naturally reference the company, year, and page numbers when mentioning important metrics.`;

    // E. Generate Response with Full Token Budget (No cut-offs)
    const chatCompletion = await groq.chat.completions.create({
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: message },
      ],
      model: "openai/gpt-oss-120b",
      temperature: 0.2,
      max_tokens: 4096,
    });

    const answer =
      chatCompletion.choices[0]?.message?.content ||
      "No response generated from model.";

    // F. Return Grounded Answer & Source Citations
    return NextResponse.json({
      answer: answer,
      sources: typedChunks,
    });
  } catch (error: unknown) {
    console.error("Chat API Error:", error);
    const errorMsg =
      error instanceof Error ? error.message : "Internal Server Error";
    return NextResponse.json({ error: errorMsg }, { status: 500 });
  }
}