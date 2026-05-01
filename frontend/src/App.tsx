import { useState, useRef, useEffect } from "react";
import ResultCard from "./components/ResultCard";
import CompareTab from "./components/CompareTab";

// ── Types ─────────────────────────────────────────────────────────────────────

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

const PIPELINE_MODES: { id: PipelineMode; label: string; desc: string }[] = [
  { id: "full",            label: "Full Pipeline",   desc: "Preprocess → Train → Validate → Test" },
  { id: "preprocess_only", label: "Preprocess Only", desc: "Tokenize & split data, then stop" },
  { id: "skip_train",      label: "Evaluate Only",   desc: "Validate & test an already-trained model" },
];

const PIPELINE_STEPS: {
  n: number;
  key: string;
  label: string;
  modes: PipelineMode[];
  detail: (model: string) => string;
}[] = [
  {
    n: 1,
    key: "preprocessing",
    label: "Preprocess",
    modes: ["full", "preprocess_only"],
    detail: () =>
      "Reads bias_clean.csv (1,733 labeled articles), strips URLs and special characters, lowercases text, and applies a stratified 60/20/20 split. Encodes three labels (Left, Center, Right) with sklearn's LabelEncoder and saves label_encoder.pkl to preprocessed_data/.",
  },
  {
    n: 2,
    key: "training",
    label: "Train",
    modes: ["full"],
    detail: (model) =>
      `Fine-tunes ${model === "bert" ? "bert-base-uncased" : "distilbert-base-uncased"} using PyTorch. Tokenizes articles up to 128 tokens, then runs 3 epochs with AdamW and a linear warmup schedule (batch size 16, ~1,041 training articles). Per-epoch loss and accuracy are logged to ClearML and streamed to the job log below.`,
  },
  {
    n: 3,
    key: "validating",
    label: "Validate",
    modes: ["full", "skip_train"],
    detail: () =>
      "Evaluates on the 346-article validation split. Prints a full sklearn classification_report with per-class precision, recall, and F1 for Left, Center, and Right — logged to ClearML. Used to catch overfitting before the final test run.",
  },
  {
    n: 4,
    key: "testing",
    label: "Test",
    modes: ["full", "skip_train"],
    detail: () =>
      "Final evaluation on the 346 held-out test articles — this split is never seen during training or validation. Produces the overall accuracy, weighted F1, precision, and recall that populate the metrics card once the job completes.",
  },
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

  const phases =
    mode === "preprocess_only" ? ["preprocessing"]
    : mode === "skip_train"    ? ["validating", "testing"]
    : PHASES.filter(p => p !== "completed");

  const current = phaseIndex(status === "completed" ? "completed" : status);

  return (
    <div className="flex items-center flex-wrap gap-y-2">
      {phases.map((phase, i) => {
        const done   = current > PHASES.indexOf(phase) || status === "completed";
        const active = PHASES[current] === phase;
        return (
          <div key={phase} className="flex items-center">
            <div className="flex flex-col items-center">
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold transition-all ${
                done ? "bg-emerald-500 text-white" : active ? "bg-slate-900 text-white ring-4 ring-slate-200" : "bg-slate-200 text-slate-400"
              }`}>
                {done && status !== "completed" ? "✓" : i + 1}
              </div>
              <span className={`text-[10px] mt-1 font-medium ${active ? "text-slate-900" : done ? "text-emerald-600" : "text-slate-400"}`}>
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
          <div key={key} className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-center">
            <p className="text-xs font-medium text-emerald-600 uppercase tracking-wider">{label}</p>
            <p className="text-2xl font-bold text-emerald-800 mt-1">{(metrics[key] * 100).toFixed(1)}%</p>
          </div>
        ) : null
      )}
    </div>
  );
}

// ── Classify Tab ──────────────────────────────────────────────────────────────

function ClassifyTab() {
  const [text, setText]     = useState("");
  const [model, setModel]   = useState<ModelKey>("bert");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState<string | null>(null);
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

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Classify a News Article</h2>
        <p className="text-slate-500 mt-1 text-sm">
          Paste any news paragraph below. The model returns <span className="font-medium text-blue-600">Left</span>,{" "}
          <span className="font-medium text-emerald-600">Center</span>, or{" "}
          <span className="font-medium text-red-600">Right</span> with a confidence score.
        </p>
      </div>

      <div className="space-y-4">
        {/* Model selector */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-slate-600 font-medium">Model:</span>
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
          <span className="ml-auto text-xs text-slate-400 hidden sm:block">
            {model === "bert" ? "91.5% accuracy · larger model" : "91.3% accuracy · ~2× faster inference"}
          </span>
        </div>

        <div className="relative">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={e => { setText(e.target.value); setResult(null); setError(null); }}
            placeholder="Paste a news article, headline, or paragraph here…"
            rows={8}
            className="w-full resize-none rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-800 placeholder-slate-400 focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-slate-100 transition-all"
          />
          <span className="absolute bottom-3 right-3 text-xs text-slate-400">
            {wordCount} {wordCount === 1 ? "word" : "words"}
          </span>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleAnalyze}
            disabled={!text.trim() || loading}
            className="flex items-center gap-2 rounded-lg bg-slate-900 px-5 py-2.5 text-sm font-semibold text-white hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
          >
            {loading ? (
              <>
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z" />
                </svg>
                Analyzing…
              </>
            ) : "Classify"}
          </button>
          {text && (
            <button
              onClick={() => { setText(""); setResult(null); setError(null); }}
              className="text-sm text-slate-400 hover:text-slate-700 transition-colors"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          <span className="font-semibold">Error: </span>{error}
        </div>
      )}

      {result && <ResultCard result={result} />}

      {!result && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-slate-400 uppercase tracking-widest">Try an example</p>
          <div className="grid gap-2">
            {EXAMPLES.map((ex, i) => (
              <button
                key={i}
                onClick={() => { setText(ex.text); setResult(null); setError(null); textareaRef.current?.focus(); }}
                className="text-left rounded-lg border border-slate-200 bg-white px-4 py-3 hover:border-slate-300 hover:bg-slate-50 transition-all group"
              >
                <span className="block text-xs font-semibold text-slate-400 mb-0.5 group-hover:text-slate-600">
                  {ex.label}
                </span>
                <span className="text-sm text-slate-600 line-clamp-2">{ex.text}</span>
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
  const [modelId, setModelId]           = useState("bert");
  const [pipelineMode, setPipelineMode] = useState<PipelineMode>("full");
  const [activeJob, setActiveJob]       = useState<Job | null>(null);
  const [pastJobs, setPastJobs]         = useState<Job[]>([]);
  const [submitting, setSubmitting]     = useState(false);
  const [error, setError]               = useState<string | null>(null);
  const logEndRef  = useRef<HTMLDivElement>(null);
  const pollRef    = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    fetch("/jobs").then(r => r.json()).then(setPastJobs).catch(() => {});
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
        body: JSON.stringify({ model_id: modelId.trim(), local: true, mode: pipelineMode }),
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

  const activeSteps = PIPELINE_STEPS.filter(s => s.modes.includes(pipelineMode));

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Training Pipeline</h2>
        <p className="text-slate-500 mt-1 text-sm">
          Fine-tune BERT or DistilBERT on the political news dataset. The pipeline runs entirely on this server.
        </p>
      </div>

      {/* Config */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 space-y-6">
        {/* Model */}
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-2">Model</p>
          <div className="grid grid-cols-2 gap-3">
            {[
              { id: "bert",       label: "BERT",       sub: "bert-base-uncased · higher accuracy" },
              { id: "distilbert", label: "DistilBERT", sub: "distilbert-base-uncased · 40% smaller" },
            ].map(m => (
              <button
                key={m.id}
                onClick={() => setModelId(m.id)}
                disabled={isRunning}
                className={`rounded-lg border-2 px-4 py-3 text-left transition-all disabled:opacity-40 ${
                  modelId === m.id
                    ? "border-slate-900 bg-slate-900 text-white"
                    : "border-slate-200 bg-white hover:border-slate-300"
                }`}
              >
                <p className={`font-bold text-sm ${modelId === m.id ? "text-white" : "text-slate-800"}`}>{m.label}</p>
                <p className={`text-xs mt-0.5 ${modelId === m.id ? "text-slate-300" : "text-slate-400"}`}>{m.sub}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Pipeline mode */}
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-2">Mode</p>
          <div className="space-y-2">
            {PIPELINE_MODES.map(pm => (
              <button
                key={pm.id}
                onClick={() => setPipelineMode(pm.id)}
                disabled={isRunning}
                className={`w-full rounded-lg border px-4 py-3 text-left flex items-center gap-3 transition-all disabled:opacity-40 ${
                  pipelineMode === pm.id ? "border-slate-900 bg-slate-50" : "border-slate-200 bg-white hover:border-slate-300"
                }`}
              >
                <div className={`w-4 h-4 rounded-full border-2 flex-shrink-0 flex items-center justify-center ${
                  pipelineMode === pm.id ? "border-slate-900" : "border-slate-300"
                }`}>
                  {pipelineMode === pm.id && <div className="w-2 h-2 rounded-full bg-slate-900" />}
                </div>
                <div>
                  <p className={`text-sm font-semibold ${pipelineMode === pm.id ? "text-slate-900" : "text-slate-700"}`}>{pm.label}</p>
                  <p className="text-xs text-slate-400 mt-0.5">{pm.desc}</p>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* What runs — detailed stage descriptions */}
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3">What runs</p>
          <div className="space-y-4">
            {activeSteps.map((step, i) => (
              <div key={step.key} className="flex gap-4">
                <div className="flex flex-col items-center flex-shrink-0">
                  <div className="w-6 h-6 rounded-full bg-slate-900 text-white text-xs font-bold flex items-center justify-center">
                    {step.n}
                  </div>
                  {i < activeSteps.length - 1 && <div className="w-0.5 flex-1 bg-slate-200 mt-1 min-h-[1.5rem]" />}
                </div>
                <div className="pb-2">
                  <p className="text-sm font-semibold text-slate-800">{step.label}</p>
                  <p className="text-sm text-slate-500 mt-0.5 leading-relaxed">{step.detail(modelId)}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <button
          onClick={handleRun}
          disabled={!modelId.trim() || isRunning || submitting}
          className="flex items-center gap-2 rounded-lg bg-slate-900 px-5 py-2.5 text-sm font-semibold text-white hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
        >
          {isRunning ? (
            <>
              <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z" />
              </svg>
              Pipeline running…
            </>
          ) : "Run Pipeline"}
        </button>

        {error && (
          <p className="text-sm text-red-600"><span className="font-semibold">Error:</span> {error}</p>
        )}
      </div>

      {/* Active job */}
      {activeJob && (
        <div className="rounded-xl border border-slate-200 bg-white p-6 space-y-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-400">Job {activeJob.job_id}</p>
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
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {activeJob.error}
            </div>
          )}

          <div>
            <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-2">Live Logs</p>
            <div className="bg-slate-950 rounded-lg p-4 h-64 overflow-y-auto font-mono text-xs text-slate-300 space-y-0.5">
              {activeJob.logs.map((line, i) => (
                <div
                  key={i}
                  className={
                    line.startsWith("[pipeline]") ? "text-blue-400 font-semibold"
                    : line.startsWith("[cmd]")     ? "text-slate-500"
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
          <div className="grid gap-2">
            {pastJobs
              .filter(j => j.job_id !== activeJob?.job_id)
              .map(job => (
                <button
                  key={job.job_id}
                  onClick={() => setActiveJob(job)}
                  className="w-full text-left rounded-lg border border-slate-200 bg-white px-4 py-3 hover:border-slate-300 hover:bg-slate-50 transition-all"
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
                      F1 {(job.metrics.f1 * 100).toFixed(1)}% · Accuracy {(job.metrics.accuracy * 100).toFixed(1)}%
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

const TABS: { id: Tab; label: string }[] = [
  { id: "classify", label: "Classify" },
  { id: "compare",  label: "Compare" },
  { id: "train",    label: "Train" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("classify");

  return (
    <div className="min-h-screen bg-white font-sans">
      <header className="border-b border-slate-200 bg-white sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-6 h-14 flex items-center gap-6">
          <div className="flex items-center gap-2 flex-shrink-0">
            <span className="text-sm font-bold text-slate-900 tracking-tight">Bias Detector</span>
            <span className="text-slate-300">·</span>
            <span className="text-xs text-slate-400 hidden sm:block">MSML605 · BERT & DistilBERT</span>
          </div>

          <nav className="flex gap-1 ml-auto">
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  tab === t.id
                    ? "bg-slate-900 text-white"
                    : "text-slate-500 hover:text-slate-800 hover:bg-slate-100"
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>

          <div className="flex items-center gap-1.5 flex-shrink-0">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-xs text-slate-400 hidden md:block">Live</span>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-10">
        {tab === "classify" && <ClassifyTab />}
        {tab === "compare"  && <CompareTab />}
        {tab === "train"    && <TrainTab />}
      </main>

      <footer className="border-t border-slate-100 py-6 mt-16">
        <p className="text-center text-xs text-slate-400">
          MSML605 Final Project · Aaryan Jadhav · Sai Malkireddy · Abhiram Metuku
        </p>
      </footer>
    </div>
  );
}
