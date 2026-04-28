import { useState, useRef } from "react";
import ResultCard from "./components/ResultCard";
import PipelineRunner from "./components/PipelineRunner";

interface PredictResponse {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  model_used: string;
}

type ModelKey = "bert" | "distilbert";
type Tab = "predict" | "pipeline";

const EXAMPLES = [
  "The administration announced sweeping new regulations today aimed at addressing climate change, drawing praise from environmental groups and sharp criticism from business leaders who called the measures overreaching.",
  "Tax cuts proposed by lawmakers would spur economic growth and keep more money in the pockets of hardworking Americans, supporters say, while critics warn the plan overwhelmingly benefits corporations.",
  "A new study published this week finds that global average temperatures rose 0.2 degrees Celsius over the past decade, continuing a long-term trend documented by researchers across multiple institutions.",
];

function PredictTab() {
  const [text, setText] = useState("");
  const [model, setModel] = useState<ModelKey>("bert");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Server error: ${res.status}`);
      }

      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const handleExample = (ex: string) => {
    setText(ex);
    setResult(null);
    setError(null);
    textareaRef.current?.focus();
  };

  return (
    <div className="space-y-6">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold text-slate-900">Detect Political Bias in News Text</h2>
        <p className="text-slate-500 max-w-xl mx-auto">
          Paste any news article or paragraph. A fine-tuned BERT model will classify it as{" "}
          <span className="text-blue-600 font-medium">left</span>,{" "}
          <span className="text-emerald-600 font-medium">center</span>, or{" "}
          <span className="text-red-600 font-medium">right</span> leaning.
        </p>
      </div>

      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-4">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-slate-600">Model:</span>
          {(["bert", "distilbert"] as ModelKey[]).map(m => (
            <button
              key={m}
              onClick={() => setModel(m)}
              className={`px-3 py-1 rounded-full text-xs font-semibold transition-colors ${
                model === m ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-500 hover:bg-slate-200"
              }`}
            >
              {m === "bert" ? "BERT" : "DistilBERT"}
            </button>
          ))}
          <span className="ml-auto text-xs text-slate-400">
            {model === "distilbert" ? "Faster, smaller" : "Higher accuracy"}
          </span>
        </div>

        <div className="relative">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={e => { setText(e.target.value); setResult(null); setError(null); }}
            placeholder="Paste a news article, headline, or any paragraph here…"
            rows={8}
            className="w-full resize-none rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-800 placeholder-slate-400 focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100 transition-all"
          />
          <span className="absolute bottom-3 right-3 text-xs text-slate-400">
            {wordCount} {wordCount === 1 ? "word" : "words"}
          </span>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleAnalyze}
            disabled={!text.trim() || loading}
            className="flex items-center gap-2 rounded-xl bg-slate-900 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
          >
            {loading ? (
              <>
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z" />
                </svg>
                Analyzing…
              </>
            ) : (
              <>
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                Analyze
              </>
            )}
          </button>
          {text && (
            <button
              onClick={() => { setText(""); setResult(null); setError(null); }}
              className="px-4 py-2.5 text-sm font-medium text-slate-500 hover:text-slate-800 transition-colors"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-700 flex items-start gap-3">
          <span className="text-red-400 mt-0.5">⚠</span>
          <div>
            <p className="font-semibold">Something went wrong</p>
            <p className="mt-0.5 text-red-600">{error}</p>
          </div>
        </div>
      )}

      {result && <ResultCard result={result} />}

      {!result && (
        <div className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Try an example</p>
          <div className="grid gap-3">
            {EXAMPLES.map((ex, i) => (
              <button
                key={i}
                onClick={() => handleExample(ex)}
                className="text-left rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600 hover:border-blue-300 hover:bg-blue-50 hover:text-slate-800 transition-all group"
              >
                <span className="block text-xs font-semibold text-slate-400 mb-1 group-hover:text-blue-500">
                  Example {i + 1}
                </span>
                <span className="line-clamp-2">{ex}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("predict");

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 font-sans">
      {/* Header */}
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 via-purple-500 to-red-500 flex items-center justify-center">
              <span className="text-white text-sm font-bold">B</span>
            </div>
            <div>
              <h1 className="text-lg font-bold text-slate-900 leading-none">Bias Detector</h1>
              <p className="text-xs text-slate-400 leading-none mt-0.5">MSML605 · Political Bias Classification</p>
            </div>
          </div>

          {/* Tab nav */}
          <div className="flex items-center gap-1 bg-slate-100 rounded-xl p-1">
            <button
              onClick={() => setTab("predict")}
              className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-all ${
                tab === "predict"
                  ? "bg-white text-slate-900 shadow-sm"
                  : "text-slate-500 hover:text-slate-700"
              }`}
            >
              Predict
            </button>
            <button
              onClick={() => setTab("pipeline")}
              className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-all ${
                tab === "pipeline"
                  ? "bg-white text-slate-900 shadow-sm"
                  : "text-slate-500 hover:text-slate-700"
              }`}
            >
              Train Pipeline
            </button>
          </div>

          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-slate-500">API connected</span>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-10">
        {tab === "predict" ? <PredictTab /> : <PipelineRunner />}
      </main>

      <footer className="border-t border-slate-200 mt-16 py-6">
        <p className="text-center text-xs text-slate-400">
          MSML605 Final Project · Bias Detection with BERT & DistilBERT
        </p>
      </footer>
    </div>
  );
}
