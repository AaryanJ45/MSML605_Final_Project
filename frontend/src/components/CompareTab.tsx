import { useState, useEffect, useRef } from "react";

type ModelKey = "bert" | "distilbert";

interface OverallMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
}

interface PerClassMetrics {
  precision: number;
  recall: number;
  f1: number;
}

interface ModelResult {
  overall: OverallMetrics;
  per_class: Record<string, PerClassMetrics>;
}

interface CompareJob {
  job_id: string;
  model_a: string;
  model_b: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  logs: string[];
  results: Record<string, ModelResult> | null;
  error: string | null;
}

const METRIC_KEYS: { key: keyof OverallMetrics; label: string }[] = [
  { key: "accuracy",  label: "Accuracy" },
  { key: "f1",        label: "F1 Score" },
  { key: "precision", label: "Precision" },
  { key: "recall",    label: "Recall" },
];

const CLASSES = ["center", "left", "right"];

const CLASS_COLORS: Record<string, string> = {
  left:   "text-blue-600",
  center: "text-emerald-600",
  right:  "text-red-600",
};

const MODEL_INFO: Record<ModelKey, { label: string; color: string; bg: string; border: string }> = {
  bert:       { label: "BERT",       color: "text-blue-700",   bg: "bg-blue-50",   border: "border-blue-200" },
  distilbert: { label: "DistilBERT", color: "text-purple-700", bg: "bg-purple-50", border: "border-purple-200" },
};

function MetricBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-slate-200 rounded-full h-1.5">
        <div
          className="h-1.5 rounded-full bg-emerald-500 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-sm font-bold text-slate-800 w-12 text-right">{pct}%</span>
    </div>
  );
}

function ModelResultColumn({
  modelKey,
  result,
}: {
  modelKey: ModelKey;
  result: ModelResult;
}) {
  const info = MODEL_INFO[modelKey];
  return (
    <div className={`rounded-2xl border ${info.border} ${info.bg} p-5 space-y-5 flex-1`}>
      <div className="flex items-center gap-2">
        <span className={`text-lg font-bold ${info.color}`}>{info.label}</span>
        {result.overall.accuracy != null && (
          <span className="ml-auto text-xs font-semibold bg-white/70 rounded-full px-2.5 py-0.5 text-slate-600 border border-slate-200">
            {Math.round(result.overall.accuracy * 100)}% acc
          </span>
        )}
      </div>

      {/* Overall metrics */}
      <div className="space-y-3">
        <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Overall</p>
        {METRIC_KEYS.map(({ key, label }) =>
          result.overall[key] != null ? (
            <div key={key}>
              <p className="text-xs text-slate-500 mb-1">{label}</p>
              <MetricBar value={result.overall[key]!} />
            </div>
          ) : null
        )}
      </div>

      {/* Per-class breakdown */}
      {Object.keys(result.per_class).length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Per Class</p>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-400">
                  <th className="text-left pb-1 font-semibold">Class</th>
                  <th className="text-right pb-1 font-semibold">Prec</th>
                  <th className="text-right pb-1 font-semibold">Rec</th>
                  <th className="text-right pb-1 font-semibold">F1</th>
                </tr>
              </thead>
              <tbody>
                {CLASSES.filter(c => result.per_class[c]).map(cls => (
                  <tr key={cls} className="border-t border-slate-200/60">
                    <td className={`py-1 font-semibold capitalize ${CLASS_COLORS[cls] ?? "text-slate-700"}`}>
                      {cls}
                    </td>
                    <td className="py-1 text-right text-slate-600">
                      {Math.round(result.per_class[cls].precision * 100)}%
                    </td>
                    <td className="py-1 text-right text-slate-600">
                      {Math.round(result.per_class[cls].recall * 100)}%
                    </td>
                    <td className="py-1 text-right font-semibold text-slate-700">
                      {Math.round(result.per_class[cls].f1 * 100)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function WinnerBanner({
  results,
  modelA,
  modelB,
}: {
  results: Record<string, ModelResult>;
  modelA: ModelKey;
  modelB: ModelKey;
}) {
  const accA = results[modelA]?.overall.accuracy ?? 0;
  const accB = results[modelB]?.overall.accuracy ?? 0;
  const winner = accA > accB ? modelA : accA < accB ? modelB : null;

  if (!winner) {
    return (
      <div className="rounded-xl bg-amber-50 border border-amber-200 px-5 py-3 text-sm text-amber-700 font-semibold text-center">
        Tie — both models achieve identical accuracy!
      </div>
    );
  }

  const info = MODEL_INFO[winner];
  const diff = Math.abs(accA - accB);
  return (
    <div className={`rounded-xl ${info.bg} border ${info.border} px-5 py-3 flex items-center justify-between`}>
      <div>
        <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Winner</p>
        <p className={`text-lg font-bold ${info.color}`}>{info.label}</p>
      </div>
      <div className="text-right">
        <p className="text-xs text-slate-400">Accuracy advantage</p>
        <p className={`text-xl font-bold ${info.color}`}>+{(diff * 100).toFixed(2)}%</p>
      </div>
    </div>
  );
}

const STATUS_LABEL: Record<string, string> = {
  queued:       "Queued",
  evaluating_a: "Evaluating Model A…",
  evaluating_b: "Evaluating Model B…",
  completed:    "Completed",
  failed:       "Failed",
};

export default function CompareTab() {
  const [modelA, setModelA] = useState<ModelKey>("bert");
  const [modelB, setModelB] = useState<ModelKey>("distilbert");
  const [job, setJob] = useState<CompareJob | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const isRunning = job !== null && job.status !== "completed" && job.status !== "failed";

  useEffect(() => {
    if (!job || job.status === "completed" || job.status === "failed") {
      if (pollRef.current) clearInterval(pollRef.current);
      return;
    }

    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`/compare/${job.job_id}`);
        const updated: CompareJob = await res.json();
        setJob(updated);
        if (updated.status === "completed" || updated.status === "failed") {
          clearInterval(pollRef.current!);
        }
      } catch {
        // keep polling on network blip
      }
    }, 2000);

    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [job?.job_id, job?.status]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [job?.logs?.length]);

  const handleCompare = async () => {
    setSubmitting(true);
    setError(null);
    setJob(null);

    try {
      const res = await fetch("/compare/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_a: modelA, model_b: modelB, local: true }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Error ${res.status}`);
      }

      const { job_id } = await res.json();
      const jobRes = await fetch(`/compare/${job_id}`);
      setJob(await jobRes.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Model Comparison</h2>
        <p className="text-slate-500 mt-1 text-sm">
          Runs <code className="bg-slate-100 px-1 rounded">test.py</code> on both models against the same 346-article
          held-out split and returns side-by-side accuracy, F1, precision, and recall — plus a per-class breakdown.
          Both models must already be trained.
        </p>
      </div>

      {/* Config card */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-6">
        <div className="grid grid-cols-2 gap-4">
          {(["bert", "distilbert"] as ModelKey[]).map((mk, idx) => {
            const isA = idx === 0;
            const current = isA ? modelA : modelB;
            const other = isA ? modelB : modelA;
            const info = MODEL_INFO[mk];
            const selected = current === mk;
            return (
              <div key={mk} className="space-y-2">
                <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">
                  Model {isA ? "A" : "B"}
                </p>
                <button
                  onClick={() => isA ? setModelA(mk) : setModelB(mk)}
                  disabled={isRunning || other === mk}
                  className={`w-full rounded-xl border-2 px-4 py-3 text-left transition-all disabled:opacity-40 ${
                    selected
                      ? `${info.border} ${info.bg}`
                      : "border-slate-200 bg-white hover:border-slate-300"
                  }`}
                >
                  <p className={`font-bold text-sm ${selected ? info.color : "text-slate-600"}`}>
                    {info.label}
                  </p>
                  <p className="text-xs text-slate-400 mt-0.5">
                    {mk === "bert" ? "Higher accuracy · Larger model" : "Faster inference · 40% smaller"}
                  </p>
                </button>
              </div>
            );
          })}
        </div>

        <button
          onClick={handleCompare}
          disabled={isRunning || submitting}
          className="flex items-center gap-2 rounded-xl bg-slate-900 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
        >
          {isRunning ? (
            <>
              <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z" />
              </svg>
              {STATUS_LABEL[job?.status ?? "queued"]}
            </>
          ) : (
            <>
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              Compare Models
            </>
          )}
        </button>

        {error && (
          <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            <p className="font-semibold">Error</p>
            <p className="mt-0.5">{error}</p>
          </div>
        )}
      </div>

      {/* Job progress + logs */}
      {job && (job.status === "evaluating_a" || job.status === "evaluating_b" || job.status === "queued" || job.status === "failed") && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-4">
          <div className="flex items-center justify-between">
            <p className="font-semibold text-slate-800">
              {MODEL_INFO[job.model_a as ModelKey].label} vs {MODEL_INFO[job.model_b as ModelKey].label}
            </p>
            <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${
              job.status === "failed"
                ? "bg-red-100 text-red-700"
                : "bg-blue-100 text-blue-700"
            }`}>
              {job.status === "evaluating_a" || job.status === "evaluating_b" || job.status === "queued"
                ? <span className="mr-1.5 h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
                : null}
              {STATUS_LABEL[job.status] ?? job.status}
            </span>
          </div>

          {/* Step indicators */}
          <div className="flex items-center gap-2">
            {(["evaluating_a", "evaluating_b"] as const).map((phase, i) => {
              const done = job.status === "completed" ||
                (phase === "evaluating_a" && job.status === "evaluating_b");
              const active = job.status === phase;
              const label = phase === "evaluating_a"
                ? MODEL_INFO[job.model_a as ModelKey].label
                : MODEL_INFO[job.model_b as ModelKey].label;
              return (
                <div key={phase} className="flex items-center gap-2">
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                    done ? "bg-emerald-500 text-white" : active ? "bg-blue-500 text-white ring-4 ring-blue-100" : "bg-slate-200 text-slate-400"
                  }`}>
                    {done ? "✓" : i + 1}
                  </div>
                  <span className={`text-xs font-medium ${active ? "text-blue-600" : done ? "text-emerald-600" : "text-slate-400"}`}>
                    {label}
                  </span>
                  {i === 0 && <div className={`h-0.5 w-8 ${done ? "bg-emerald-400" : "bg-slate-200"}`} />}
                </div>
              );
            })}
          </div>

          {job.status === "failed" && job.error && (
            <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {job.error}
            </div>
          )}

          <div className="bg-slate-950 rounded-xl p-4 h-48 overflow-y-auto font-mono text-xs text-slate-300 space-y-0.5">
            {job.logs.map((line, i) => (
              <div
                key={i}
                className={
                  line.startsWith("[compare]") ? "text-blue-400 font-semibold"
                  : line.startsWith("[cmd") ? "text-slate-500"
                  : "text-slate-300"
                }
              >
                {line}
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        </div>
      )}

      {/* Results */}
      {job?.status === "completed" && job.results && (
        <div className="space-y-5">
          <WinnerBanner
            results={job.results}
            modelA={job.model_a as ModelKey}
            modelB={job.model_b as ModelKey}
          />

          <div className="flex gap-4">
            <ModelResultColumn
              modelKey={job.model_a as ModelKey}
              result={job.results[job.model_a]}
            />
            <ModelResultColumn
              modelKey={job.model_b as ModelKey}
              result={job.results[job.model_b]}
            />
          </div>

          {/* Context note */}
          <p className="text-xs text-slate-400">
            Both models are fine-tuned for 3 epochs on 1,041 training articles. DistilBERT is ~40% smaller and
            ~2× faster at inference while retaining 97% of BERT's language understanding through knowledge distillation.
          </p>
        </div>
      )}
    </div>
  );
}
