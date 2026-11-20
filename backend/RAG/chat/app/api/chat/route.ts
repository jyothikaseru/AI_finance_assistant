import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";
import Groq from "groq-sdk";

// ---------------------------------------------------------------------------
// 1. Clients & Configuration
// ---------------------------------------------------------------------------

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const groqApiKey = process.env.GROQ_API_KEY!;

const supabase = createClient(supabaseUrl, supabaseKey);
const groq = new Groq({ apiKey: groqApiKey });

// ---------------------------------------------------------------------------
// 2. TypeScript Interfaces
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 3. Local BAAI/BGE Embedding Pipeline (768-dim normalized)
// ---------------------------------------------------------------------------

type EmbedderPipeline = (
  text: string,
  options?: { pooling: string; normalize: boolean }
) => Promise<{ data: number[] | Float32Array }>;

let embedder: EmbedderPipeline | null = null;

async function getBAAIEmbedding(text: string): Promise<number[]> {
  try {
    if (!embedder) {
      const { pipeline, env } = await import("@xenova/transformers");
      env.allowLocalModels = false; // Fetch ONNX model from Hugging Face hub
      embedder = (await pipeline(
        "feature-extraction",
        "Xenova/bge-base-en-v1.5"
      )) as unknown as EmbedderPipeline;
    }

    const output = await embedder(text, { pooling: "mean", normalize: true });
    return Array.from(output.data);
  } catch (err: unknown) {
    console.error("Local embedding fallback:", err);
    // Return empty array / zeros so tsvector full-text search handles query
    return new Array(768).fill(0);
  }
}

// ---------------------------------------------------------------------------
// 4. Metadata Filter Extraction (Single-Company vs. Comparison Detection)
// ---------------------------------------------------------------------------

function extractFilterMetadata(
  query: string,
  explicitTicker?: string,
  explicitYear?: string | number
) {
  let resolvedTicker: string | null = explicitTicker || null;
  let resolvedYear: number | null = explicitYear ? parseInt(String(explicitYear)) : null;

  if (!resolvedTicker) {
    const matchedTickers: string[] = [];
    if (/\b(apple|aapl)\b/i.test(query)) matchedTickers.push("AAPL");
    if (/\b(microsoft|msft)\b/i.test(query)) matchedTickers.push("MSFT");
    if (/\b(infosys|infy)\b/i.test(query)) matchedTickers.push("INFY");
    if (/\b(reliance|ril)\b/i.test(query)) matchedTickers.push("RELIANCE");
    if (/\b(tcs|tata\s+consultancy)\b/i.test(query)) matchedTickers.push("TCS");

    // Only apply single-ticker filter if exactly ONE company is mentioned.
    // If multiple are mentioned (e.g. "Compare Apple and Microsoft"), keep null to search both.
    if (matchedTickers.length === 1) {
      resolvedTicker = matchedTickers[0];
    }
  }

  if (!resolvedYear) {
    const yearMatch = query.match(/\b(202[1-6])\b/);
    if (yearMatch) {
      resolvedYear = parseInt(yearMatch[1]);
    }
  }

  return { resolvedTicker, resolvedYear };
}

// ---------------------------------------------------------------------------
// 5. Main Route Handler (POST /api/chat)
// ---------------------------------------------------------------------------

export async function POST(req: NextRequest) {
  try {
    const { message, ticker, fiscalYear } = await req.json();

    if (!message || typeof message !== "string") {
      return NextResponse.json({ error: "Message is required" }, { status: 400 });
    }

    // A. Intelligent Metadata Filter Extraction
    const { resolvedTicker, resolvedYear } = extractFilterMetadata(
      message,
      ticker,
      fiscalYear
    );

    // B. Generate 768-dim Query Vector
    const queryEmbedding = await getBAAIEmbedding(message);

    // C. Hybrid Retrieval in PostgreSQL (HNSW Vector + tsvector Full-Text)
    const { data: chunks, error: rpcError } = await supabase.rpc(
      "match_financial_chunks",
      {
        query_text: message,
        query_embedding: queryEmbedding,
        match_count: 8,
        filter_ticker: resolvedTicker,
        filter_year: resolvedYear,
        filter_type: null,
      }
    );

    if (rpcError) {
      console.error("Supabase RPC Error:", rpcError);
      return NextResponse.json(
        { error: `Database retrieval failed: ${rpcError.message}` },
        { status: 500 }
      );
    }

    const typedChunks: FinancialChunk[] = chunks || [];

    // D. Construct Grounded Financial Context
    const contextText = typedChunks
      .map(
        (c, i) =>
          `[Source ${i + 1} | Company: ${c.ticker} | Year: ${c.fiscal_year} | Section: ${
            c.section || "General"
          } | Page: ${c.page_start}]\n${c.content}`
      )
      .join("\n\n---\n\n");

    // E. Financial Analyst System Prompt with Strict Grounding Guardrails
    const systemPrompt = `You are FinRAG, an elite financial intelligence assistant specializing in corporate Annual Reports (10-K and SEBI Integrated Reports).

Context from Indexed Financial Reports:
${contextText || "No matching reports found in database."}

Instructions:
1. Tone & Style: Respond like a professional senior financial consultant—concise, analytical, and authoritative.
2. Grounding Guardrails: Base your figures and conclusions strictly on the provided context. If a requested metric or company is missing from the context, state: "I don't have that specific data in the indexed annual reports."
3. Formatting: Present key metrics and ratios in clean visual tables or bullet points, followed by a succinct analytical explanation.
4. Citations: Reference the company, fiscal year, and exact page numbers when stating financial figures.`;

    // F. Resilient LLM Inference on Groq (using models provisioned on your Groq API key)
    let answer = "";
    const groqCandidateModels = [
      "openai/gpt-oss-120b",
      "openai/gpt-oss-20b",
      "qwen/qwen3.8-27b",
    ];

    let lastError: Error | unknown = null;

    for (const modelName of groqCandidateModels) {
      try {
        console.log(`[Groq] Attempting generation with model: ${modelName}`);
        const chatCompletion = await groq.chat.completions.create({
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: message },
          ],
          model: modelName,
          temperature: 0.2,
          max_tokens: 4096,
        });

        const outputText = chatCompletion.choices[0]?.message?.content;
        if (outputText && outputText.trim()) {
          answer = outputText;
          console.log(`[Groq] Successfully generated response with ${modelName}`);
          break;
        }
      } catch (modelErr: unknown) {
        lastError = modelErr;
        console.warn(`[Groq] Model ${modelName} failed:`, modelErr);
      }
    }

    if (!answer) {
      const errDetail = lastError instanceof Error ? lastError.message : String(lastError || "");
      answer = `Unable to generate LLM response. Detail: ${errDetail}`;
    }

    // G. Return Grounded Answer & Source Citations
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
