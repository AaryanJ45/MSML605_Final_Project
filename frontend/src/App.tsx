import { useState, useRef, useEffect } from "react";
import ResultCard from "./components/ResultCard";
import CompareTab from "./components/CompareTab";

// ── Types ────────────────────────────────────────────────────────────────────

interface PredictResponse {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  model_used: string;
}

type ModelKey = "bert" | "distilbert";
type Tab = "classify" | "compare" | "train";
type PipelineMode = "full" | "preprocess_only" | "skip_train";

interface Job {
  job_id: string;
  model_id: string;
  mode?: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  logs: string[];
  metrics: Record<string, number> | null;
  error: string | null;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const EXAMPLES = [
  {
    label: "Climate policy",
    text: "The administration announced sweeping new regulations today aimed at addressing climate change, drawing praise from environmental groups and sharp criticism from business leaders who called the measures overreaching.",
  },
  {
    label: "Tax policy",
    text: "Tax cuts proposed by lawmakers would spur economic growth and keep more money in the pockets of hardworking Americans, supporters say, while critics warn the plan overwhelmingly benefits corporations.",
  },
  {
    label: "Scientific finding",
    text: "A new study published this week finds that global average temperatures rose 0.2 degrees Celsius over the past decade, continuing a long-term trend documented by researchers across multiple institutions.",
  },
  {
    label: "Border security",
    text: "Congress debated new border security legislation Wednesday, with Republicans demanding stricter enforcement measures while Democrats pushed for expanded pathways to legal status for undocumented workers.",
  },
];

const PHASES = ["preprocessing", "training", "validating", "testing", "completed"];

const PHASE_LABEL: Record<string, string> = {
  queued:        "Queued",
  preprocessing: "Preprocessing",
  training:      "Training",
  validating:    "Validating",
  testing:       "Testing",
  completed:     "Completed",
  failed:        "Failed",
};

const POPULAR_MODELS = [
  { id: "bert",       label: "BERT",       desc: "Higher accuracy · Google 2018" },
  { id: "distilbert", label: "DistilBERT", desc: "Faster · 40% smaller · HuggingFace" },
];

const PIPELINE_MODES: { id: PipelineMode; label: string; desc: string }[] = [
  { id: "full",            label: "Full Pipeline",    desc: "Preprocess → Train → Validate → Test" },
  { id: "preprocess_only", label: "Preprocess Only",  desc: "Tokenize & split data, then stop" },
  { id: "skip_train",      label: "Evaluate Only",    desc: "Skip training — validate & test existing model" },
];

// ── Shared helpers ────────────────────────────────────────────────────────────

function phaseIndex(status: string): number {
  return PHASES.indexOf(status);
}

function StatusBadge({ status }: { status: string }) {
  const cls: Record<string, string> = {
    queued:        "bg-slate-100 text-slate-500",
    preprocessing: "bg-amber-100 text-amber-700",
    training:      "bg-blue-100 text-blue-700",
    validating:    "bg-purple-100 text-purple-700",
    testing:       "bg-orange-100 text-orange-700",
    completed:     "bg-emerald-100 text-emerald-700",
    failed:        "bg-red-100 text-red-700",
  };
  const pulse = ["training", "preprocessing", "validating", "testing"].includes(status);
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${cls[status] ?? "bg-slate-100 text-slate-500"}`}>
      {pulse && <span className="mr-1.5 h-1.5 w-1.5 rounded-full bg-current animate-pulse" />}
      {PHASE_LABEL[status] ?? status}
    </span>
  );
}

function PhaseBar({ status, mode }: { status: string; mode?: string }) {
  if (status === "failed") {
    return <div className="flex items-center gap-2 text-red-500 text-sm font-medium"><span>✗</span> Pipeline failed</div>;
  }

  // For preprocess_only mode only show one step
  const phases = mode === "preprocess_only"
    ? ["preprocessing"]
    : mode === "skip_train"
    ? ["validating", "testing"]
    : PHASES.filter(p => p !== "completed");

  const current = phaseIndex(status === "completed" ? "completed" : status);

  return (
    <div className="flex items-center gap-0 flex-wrap gap-y-2">
      {phases.map((phase, i) => {
        const done = current > PHASES.indexOf(phase) || status === "completed";
        const active = PHASES[current] === phase;
        return (
          <div key={phase} className="flex items-center">
            <div className="flex flex-col items-center">
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold transition-all ${
                done ? "bg-emerald-500 text-white" : active ? "bg-blue-500 text-white ring-4 ring-blue-100" : "bg-slate-200 text-slate-400"
              }`}>
                {done && status !== "completed" ? "✓" : i + 1}
              </div>
              <span className={`text-[10px] mt-1 font-medium ${active ? "text-blue-600" : done ? "text-emerald-600" : "text-slate-400"}`}>
                {PHASE_LABEL[phase]}
              </span>
            </div>
            {i < phases.length - 1 && (
              <div className={`h-0.5 w-8 mx-1 mb-4 transition-all ${done ? "bg-emerald-400" : "bg-slate-200"}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}

function MetricsCard({ metrics }: { metrics: Record<string, number> }) {
  const items = [
    { key: "accuracy",  label: "Accuracy" },
    { key: "f1",        label: "F1 Score" },
    { key: "precision", label: "Precision" },
    { key: "recall",    label: "Recall" },
  ];
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      {items.map(({ key, label }) =>
        metrics[key] != null ? (
          <div key={key} className="rounded-xl bg-emerald-50 border border-emerald-200 p-3 text-center">
            <p className="text-xs font-semibold text-emerald-500 uppercase tracking-wider">{label}</p>
            <p className="text-2xl font-bold text-emerald-700 mt-1">{(metrics[key] * 100).toFixed(1)}%</p>
          </div>
        ) : null
      )}
    </div>
  );
}

// ── Classify Tab ──────────────────────────────────────────────────────────────

function ClassifyTab() {
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
    <div className="space-y-8">
      {/* Header */}
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 via-purple-500 to-red-500 flex items-center justify-center text-white font-bold text-lg">
            B
          </div>
          <div>
            <h2 className="text-2xl font-bold text-slate-900">Classify News Article Bias</h2>
            <p className="text-sm text-slate-500">Real-time political lean detection using fine-tuned transformers</p>
          </div>
        </div>

        {/* Explainer */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl px-5 py-4 text-sm text-slate-600 space-y-2">
          <p className="font-semibold text-slate-700">How it works</p>
          <p>
            Paste any news article or paragraph. The model runs the text through a fine-tuned
            BERT or DistilBERT transformer — each trained on 1,041 labeled political-news articles
            from a 1,733-article Kaggle dataset — and returns a{" "}
            <span className="font-semibold text-blue-600">Left</span>,{" "}
            <span className="font-semibold text-emerald-600">Center</span>, or{" "}
            <span className="font-semibold text-red-600">Right</span> label with a confidence score
            and full probability distribution over all three classes.
          </p>
          <div className="grid grid-cols-3 gap-3 pt-1">
            {[
              { label: "Left", color: "text-blue-600", bg: "bg-blue-50 border-blue-200", desc: "Progressive framing, emphasizes equity & social policy" },
              { label: "Center", color: "text-emerald-600", bg: "bg-emerald-50 border-emerald-200", desc: "Balanced or neutral coverage without strong lean" },
              { label: "Right", color: "text-red-600", bg: "bg-red-50 border-red-200", desc: "Conservative framing, emphasizes traditional values & market policy" },
            ].map(c => (
              <div key={c.label} className={`rounded-lg border px-3 py-2 ${c.bg}`}>
                <p className={`font-bold text-sm ${c.color}`}>{c.label}</p>
                <p className="text-xs text-slate-500 mt-0.5">{c.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Input card */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-4">
        {/* Model selector */}
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-slate-600">Model:</span>
          {(["bert", "distilbert"] as ModelKey[]).map(m => (
            <button
              key={m}
              onClick={() => { setModel(m); setResult(null); }}
              className={`px-3 py-1 rounded-full text-xs font-semibold transition-colors ${
                model === m ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-500 hover:bg-slate-200"
              }`}
            >
              {m === "bert" ? "BERT" : "DistilBERT"}
            </button>
          ))}
          <span className="ml-auto text-xs text-slate-400">
            {model === "bert"
              ? "91.5% accuracy · Best for high-stakes classification"
              : "91.3% accuracy · 2× faster inference · ideal for production"}
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
                Classify
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
                onClick={() => handleExample(ex.text)}
                className="text-left rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600 hover:border-blue-300 hover:bg-blue-50 hover:text-slate-800 transition-all group"
              >
                <span className="block text-xs font-semibold text-slate-400 mb-1 group-hover:text-blue-500">
                  {ex.label}
                </span>
                <span className="line-clamp-2">{ex.text}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Train Tab ─────────────────────────────────────────────────────────────────

function TrainTab() {
  const [modelId, setModelId] = useState("bert");
  const [local, setLocal] = useState(true);
  const [pipelineMode, setPipelineMode] = useState<PipelineMode>("full");
  const [activeJob, setActiveJob] = useState<Job | null>(null);
  const [pastJobs, setPastJobs] = useState<Job[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    fetch("/jobs")
      .then(r => r.json())
      .then(setPastJobs)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!activeJob || activeJob.status === "completed" || activeJob.status === "failed") {
      if (pollRef.current) clearInterval(pollRef.current);
      return;
    }

    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`/jobs/${activeJob.job_id}`);
        const updated: Job = await res.json();
        setActiveJob(updated);
        if (updated.status === "completed" || updated.status === "failed") {
          clearInterval(pollRef.current!);
          fetch("/jobs").then(r => r.json()).then(setPastJobs).catch(() => {});
        }
      } catch { /* keep polling */ }
    }, 2000);

    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [activeJob?.job_id, activeJob?.status]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeJob?.logs?.length]);

  const isRunning = activeJob !== null && activeJob.status !== "completed" && activeJob.status !== "failed";

  const handleRun = async () => {
    if (!modelId.trim()) return;
    setSubmitting(true);
    setError(null);
    setActiveJob(null);

    try {
      const res = await fetch("/jobs/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId.trim(), local, mode: pipelineMode }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Error ${res.status}`);
      }

      const { job_id } = await res.json();
      const jobRes = await fetch(`/jobs/${job_id}`);
      setActiveJob(await jobRes.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center text-white font-bold text-sm">
            ▶
          </div>
          <div>
            <h2 className="text-2xl font-bold text-slate-900">Training Pipeline</h2>
            <p className="text-sm text-slate-500">Run the end-to-end MLOps pipeline — preprocess, train, validate, and test</p>
          </div>
        </div>

        {/* Pipeline explainer */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl px-5 py-4 text-sm text-slate-600 space-y-3">
          <p className="font-semibold text-slate-700">Pipeline Stages</p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { step: "1", label: "Preprocess", desc: "Tokenize bias_clean.csv, split 60/20/20 into train/val/test CSVs, encode labels, upload to S3" },
              { step: "2", label: "Train",      desc: "Fine-tune BERT or DistilBERT for 2 epochs using PyTorch, log loss/accuracy to ClearML" },
              { step: "3", label: "Validate",   desc: "Evaluate on the validation set (346 articles), log precision / recall / F1 per class" },
              { step: "4", label: "Test",       desc: "Final evaluation on held-out test set, generate classification report, save to S3" },
            ].map(s => (
              <div key={s.step} className="bg-white border border-slate-200 rounded-lg px-3 py-2">
                <div className="flex items-center gap-2 mb-1">
                  <span className="w-5 h-5 rounded-full bg-slate-900 text-white text-xs font-bold flex items-center justify-center">{s.step}</span>
                  <span className="font-semibold text-slate-800 text-xs">{s.label}</span>
                </div>
                <p className="text-xs text-slate-500">{s.desc}</p>
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-400">
            ClearML tracks all metrics. Model artifacts are automatically saved to <code className="bg-slate-200 px-1 rounded">saved_models/</code> and synced to the configured S3 bucket after every run.
          </p>
        </div>
      </div>

      {/* Config card */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-5">
        <div>
          <h3 className="text-base font-bold text-slate-900">Configure Run</h3>
          <p className="text-sm text-slate-500 mt-0.5">Choose a model, pipeline mode, and data source.</p>
        </div>

        {/* Model selector */}
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-2">Model</p>
          <div className="grid grid-cols-2 gap-3">
            {POPULAR_MODELS.map(m => (
              <button
                key={m.id}
                onClick={() => setModelId(m.id)}
                disabled={isRunning}
                className={`rounded-xl border-2 px-4 py-3 text-left transition-all disabled:opacity-40 ${
                  modelId === m.id
                    ? "border-slate-900 bg-slate-900 text-white"
                    : "border-slate-200 bg-white hover:border-slate-300"
                }`}
              >
                <p className={`font-bold text-sm ${modelId === m.id ? "text-white" : "text-slate-800"}`}>{m.label}</p>
                <p className={`text-xs mt-0.5 ${modelId === m.id ? "text-slate-300" : "text-slate-400"}`}>{m.desc}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Pipeline mode */}
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-2">Pipeline Mode</p>
          <div className="space-y-2">
            {PIPELINE_MODES.map(pm => (
              <button
                key={pm.id}
                onClick={() => setPipelineMode(pm.id)}
                disabled={isRunning}
                className={`w-full rounded-xl border px-4 py-3 text-left flex items-center gap-3 transition-all disabled:opacity-40 ${
                  pipelineMode === pm.id
                    ? "border-blue-300 bg-blue-50"
                    : "border-slate-200 bg-white hover:border-slate-300"
                }`}
              >
                <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center flex-shrink-0 ${
                  pipelineMode === pm.id ? "border-blue-500" : "border-slate-300"
                }`}>
                  {pipelineMode === pm.id && <div className="w-2 h-2 rounded-full bg-blue-500" />}
                </div>
                <div>
                  <p className={`text-sm font-semibold ${pipelineMode === pm.id ? "text-blue-700" : "text-slate-700"}`}>{pm.label}</p>
                  <p className="text-xs text-slate-400 mt-0.5">{pm.desc}</p>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Data source toggle */}
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-slate-600">Data source:</span>
          {(["local", "s3"] as const).map(src => (
            <button
              key={src}
              onClick={() => setLocal(src === "local")}
              disabled={isRunning}
              className={`px-3 py-1 rounded-full text-xs font-semibold transition-colors disabled:opacity-40 ${
                (src === "local") === local
                  ? "bg-slate-900 text-white"
                  : "bg-slate-100 text-slate-500 hover:bg-slate-200"
              }`}
            >
              {src === "local" ? "Local" : "S3 Bucket"}
            </button>
          ))}
          <span className="text-xs text-slate-400">
            {local ? "Reads bias_clean.csv from disk" : "Pulls/pushes from configured S3 bucket"}
          </span>
        </div>

        <button
          onClick={handleRun}
          disabled={!modelId.trim() || isRunning || submitting}
          className="flex items-center gap-2 rounded-xl bg-slate-900 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
        >
          {isRunning ? (
            <>
              <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z" />
              </svg>
              Pipeline running…
            </>
          ) : (
            <>
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Run Pipeline
            </>
          )}
        </button>

        {error && (
          <p className="text-sm text-red-600 flex items-center gap-2"><span>⚠</span> {error}</p>
        )}
      </div>

      {/* Active job */}
      {activeJob && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Job {activeJob.job_id}</p>
              <p className="font-bold text-slate-900 mt-0.5">{activeJob.model_id}</p>
              {activeJob.mode && (
                <p className="text-xs text-slate-400 mt-0.5">
                  {PIPELINE_MODES.find(m => m.id === activeJob.mode)?.label ?? activeJob.mode}
                </p>
              )}
            </div>
            <StatusBadge status={activeJob.status} />
          </div>

          <PhaseBar status={activeJob.status} mode={activeJob.mode} />

          {activeJob.status === "completed" && activeJob.metrics && (
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3">Test Metrics</p>
              <MetricsCard metrics={activeJob.metrics} />
            </div>
          )}

          {activeJob.status === "failed" && activeJob.error && (
            <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {activeJob.error}
            </div>
          )}

          <div>
            <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-2">Live Logs</p>
            <div className="bg-slate-950 rounded-xl p-4 h-64 overflow-y-auto font-mono text-xs text-slate-300 space-y-0.5">
              {activeJob.logs.map((line, i) => (
                <div
                  key={i}
                  className={
                    line.startsWith("[pipeline]") ? "text-blue-400 font-semibold"
                    : line.startsWith("[cmd]") ? "text-slate-500"
                    : "text-slate-300"
                  }
                >
                  {line}
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>
        </div>
      )}

      {/* Past jobs */}
      {pastJobs.filter(j => j.job_id !== activeJob?.job_id).length > 0 && (
        <div className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Previous Jobs</p>
          <div className="grid gap-3">
            {pastJobs
              .filter(j => j.job_id !== activeJob?.job_id)
              .map(job => (
                <button
                  key={job.job_id}
                  onClick={() => setActiveJob(job)}
                  className="w-full text-left rounded-xl border border-slate-200 bg-white p-4 hover:border-blue-300 hover:bg-blue-50 transition-all"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-semibold text-slate-800 text-sm">{job.model_id}</p>
                      <p className="text-xs text-slate-400 mt-0.5">
                        {new Date(job.started_at).toLocaleString()} · Job {job.job_id}
                        {job.mode && ` · ${PIPELINE_MODES.find(m => m.id === job.mode)?.label ?? job.mode}`}
                      </p>
                    </div>
                    <StatusBadge status={job.status} />
                  </div>
                  {job.metrics && (
                    <p className="text-xs text-emerald-600 mt-2 font-medium">
                      F1: {(job.metrics.f1 * 100).toFixed(1)}% · Accuracy: {(job.metrics.accuracy * 100).toFixed(1)}%
                    </p>
                  )}
                </button>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── App Shell ─────────────────────────────────────────────────────────────────

const TAB_CONFIG: { id: Tab; label: string; shortLabel: string }[] = [
  { id: "classify", label: "Classify",   shortLabel: "Classify" },
  { id: "compare",  label: "Compare",    shortLabel: "Compare" },
  { id: "train",    label: "Train",      shortLabel: "Train" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("classify");

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 font-sans">
      {/* Header */}
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500 via-purple-500 to-red-500 flex items-center justify-center flex-shrink-0">
              <span className="text-white text-sm font-bold">B</span>
            </div>
            <div>
              <h1 className="text-base font-bold text-slate-900 leading-none">Bias Detector</h1>
              <p className="text-xs text-slate-400 leading-none mt-0.5">MSML605 · Containerized MLOps Pipeline · BERT & DistilBERT</p>
            </div>
          </div>

          {/* Tab nav */}
          <div className="flex items-center gap-1 bg-slate-100 rounded-xl p-1">
            {TAB_CONFIG.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-all ${
                  tab === t.id
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-500 hover:text-slate-700"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2 flex-shrink-0">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-slate-500 hidden sm:block">API connected</span>
          </div>
        </div>
      </header>

      {/* Hero banner */}
      <div className="bg-gradient-to-r from-slate-900 via-blue-950 to-slate-900 text-white">
        <div className="max-w-5xl mx-auto px-6 py-8">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 items-center">
            <div className="sm:col-span-2 space-y-2">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-xs font-semibold bg-blue-500/20 border border-blue-400/30 text-blue-300 rounded-full px-3 py-0.5">MSML605 Final Project</span>
                <span className="text-xs font-semibold bg-white/10 border border-white/20 text-slate-300 rounded-full px-3 py-0.5">AWS ECS · Docker · ClearML · FastAPI</span>
              </div>
              <h2 className="text-xl font-bold leading-snug">
                Containerized MLOps Pipeline<br />
                <span className="text-blue-300">for Political Bias Detection in News Articles</span>
              </h2>
              <p className="text-sm text-slate-400 max-w-lg">
                An end-to-end platform that trains, evaluates, and serves transformer-based text classifiers for political bias detection — all triggered from this UI. Fine-tuned BERT and DistilBERT achieve <span className="text-white font-semibold">91–92% accuracy</span> on a 1,733-article held-out test set.
              </p>
              <p className="text-xs text-slate-500">Aaryan Jadhav · Sai Malkireddy · Abhiram Metuku</p>
            </div>

            {/* Quick stats */}
            <div className="grid grid-cols-2 gap-3">
              {[
                { val: "1,733", label: "Articles", sub: "bias_clean.csv" },
                { val: "91.5%", label: "BERT Acc",  sub: "Weighted F1" },
                { val: "91.3%", label: "DistilBERT", sub: "Weighted F1" },
                { val: "3",     label: "Classes",   sub: "L · C · R" },
              ].map(s => (
                <div key={s.label} className="bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-center">
                  <p className="text-xl font-bold text-white">{s.val}</p>
                  <p className="text-xs font-semibold text-slate-300">{s.label}</p>
                  <p className="text-xs text-slate-500">{s.sub}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <main className="max-w-5xl mx-auto px-6 py-10">
        {tab === "classify" && <ClassifyTab />}
        {tab === "compare"  && <CompareTab />}
        {tab === "train"    && <TrainTab />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 mt-16 py-8 bg-white">
        <div className="max-w-5xl mx-auto px-6">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 text-xs text-slate-500">
            <div>
              <p className="font-semibold text-slate-700 mb-1">Architecture</p>
              <p>Kaggle Dataset → AWS S3 → ECS Preprocessing → ECS BERT/DistilBERT Training → ClearML → FastAPI → React UI</p>
            </div>
            <div>
              <p className="font-semibold text-slate-700 mb-1">API Endpoints</p>
              <ul className="space-y-0.5 font-mono">
                <li>POST /predict — inference</li>
                <li>POST /jobs/run — start training</li>
                <li>GET /jobs/:id — poll job</li>
                <li>POST /compare/run — compare models</li>
              </ul>
            </div>
            <div>
              <p className="font-semibold text-slate-700 mb-1">Future Scope</p>
              <ul className="space-y-0.5">
                <li>Browser extension for real-time bias labeling</li>
                <li>Continuous learning via scheduled ECS tasks</li>
                <li>Multi-lingual support with mBERT / XLM-R</li>
              </ul>
            </div>
          </div>
          <p className="text-center text-xs text-slate-400 mt-6 border-t border-slate-100 pt-4">
            MSML605 Final Project · Political Bias Detection with BERT & DistilBERT · Aaryan Jadhav · Sai Malkireddy · Abhiram Metuku
          </p>
        </div>
      </footer>
    </div>
  );
}
