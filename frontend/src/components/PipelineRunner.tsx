import { useState, useEffect, useRef } from "react";

interface Job {
  job_id: string;
  model_id: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  logs: string[];
  metrics: Record<string, number> | null;
  error: string | null;
}

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
  { id: "bert-base-uncased",             label: "BERT" },
  { id: "distilbert-base-uncased",       label: "DistilBERT" },
  { id: "roberta-base",                  label: "RoBERTa" },
  { id: "albert-base-v2",               label: "ALBERT" },
  { id: "microsoft/deberta-v3-small",   label: "DeBERTa" },
];

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
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${cls[status] ?? "bg-slate-100 text-slate-500"}`}>
      {status === "training" || status === "preprocessing" || status === "validating" || status === "testing"
        ? <span className="mr-1.5 h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
        : null}
      {PHASE_LABEL[status] ?? status}
    </span>
  );
}

function PhaseBar({ status }: { status: string }) {
  if (status === "failed") {
    return (
      <div className="flex items-center gap-2 text-red-500 text-sm font-medium">
        <span>✗</span> Pipeline failed
      </div>
    );
  }

  const current = phaseIndex(status === "completed" ? "completed" : status);

  return (
    <div className="flex items-center gap-0">
      {PHASES.filter(p => p !== "completed").map((phase, i) => {
        const done = current > i || status === "completed";
        const active = PHASES[current] === phase;
        return (
          <div key={phase} className="flex items-center">
            <div className={`flex flex-col items-center`}>
              <div
                className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold transition-all ${
                  done || status === "completed"
                    ? "bg-emerald-500 text-white"
                    : active
                    ? "bg-blue-500 text-white ring-4 ring-blue-100"
                    : "bg-slate-200 text-slate-400"
                }`}
              >
                {done && status !== "completed" ? "✓" : i + 1}
              </div>
              <span className={`text-[10px] mt-1 font-medium ${active ? "text-blue-600" : done ? "text-emerald-600" : "text-slate-400"}`}>
                {PHASE_LABEL[phase]}
              </span>
            </div>
            {i < PHASES.filter(p => p !== "completed").length - 1 && (
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
            <p className="text-2xl font-bold text-emerald-700 mt-1">
              {(metrics[key] * 100).toFixed(1)}%
            </p>
          </div>
        ) : null
      )}
    </div>
  );
}

function JobCard({ job, onSelect }: { job: Job; onSelect: (j: Job) => void }) {
  return (
    <button
      onClick={() => onSelect(job)}
      className="w-full text-left rounded-xl border border-slate-200 bg-white p-4 hover:border-blue-300 hover:bg-blue-50 transition-all"
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="font-semibold text-slate-800 text-sm">{job.model_id}</p>
          <p className="text-xs text-slate-400 mt-0.5">
            {new Date(job.started_at).toLocaleString()} · Job {job.job_id}
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
  );
}

export default function PipelineRunner() {
  const [modelId, setModelId] = useState("bert-base-uncased");
  const [local, setLocal] = useState(true);
  const [activeJob, setActiveJob] = useState<Job | null>(null);
  const [pastJobs, setPastJobs] = useState<Job[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load past jobs on mount
  useEffect(() => {
    fetch("/jobs")
      .then(r => r.json())
      .then(setPastJobs)
      .catch(() => {});
  }, []);

  // Poll active job
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
          // Refresh past jobs list
          fetch("/jobs").then(r => r.json()).then(setPastJobs).catch(() => {});
        }
      } catch {
        // network hiccup — keep polling
      }
    }, 2000);

    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [activeJob?.job_id, activeJob?.status]);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeJob?.logs?.length]);

  const handleRun = async () => {
    if (!modelId.trim()) return;
    setSubmitting(true);
    setError(null);
    setActiveJob(null);

    try {
      const res = await fetch("/jobs/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId.trim(), local }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Error ${res.status}`);
      }

      const { job_id } = await res.json();
      // Immediately fetch the full job object to start polling
      const jobRes = await fetch(`/jobs/${job_id}`);
      setActiveJob(await jobRes.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setSubmitting(false);
    }
  };

  const isRunning = activeJob !== null &&
    activeJob.status !== "completed" &&
    activeJob.status !== "failed";

  return (
    <div className="space-y-6">
      {/* Config card */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-5">
        <div>
          <h2 className="text-lg font-bold text-slate-900">Run Training Pipeline</h2>
          <p className="text-sm text-slate-500 mt-0.5">
            Pick any HuggingFace base model. The full pipeline runs on this server:
            preprocess → train → validate → test.
          </p>
        </div>

        {/* Popular model chips */}
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-2">
            Popular models
          </p>
          <div className="flex flex-wrap gap-2">
            {POPULAR_MODELS.map(m => (
              <button
                key={m.id}
                onClick={() => setModelId(m.id)}
                disabled={isRunning}
                className={`px-3 py-1.5 rounded-full text-xs font-semibold transition-colors disabled:opacity-40 ${
                  modelId === m.id
                    ? "bg-slate-900 text-white"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* Custom model ID input */}
        <div>
          <label className="block text-xs font-semibold uppercase tracking-widest text-slate-400 mb-1.5">
            HuggingFace Model ID
          </label>
          <input
            type="text"
            value={modelId}
            onChange={e => setModelId(e.target.value)}
            disabled={isRunning}
            placeholder="e.g. roberta-base or microsoft/deberta-v3-small"
            className="w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100 disabled:opacity-50 transition-all"
          />
        </div>

        {/* Local vs S3 toggle */}
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-slate-600">Data source:</span>
          <button
            onClick={() => setLocal(true)}
            disabled={isRunning}
            className={`px-3 py-1 rounded-full text-xs font-semibold transition-colors disabled:opacity-40 ${local ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-500 hover:bg-slate-200"}`}
          >
            Local
          </button>
          <button
            onClick={() => setLocal(false)}
            disabled={isRunning}
            className={`px-3 py-1 rounded-full text-xs font-semibold transition-colors disabled:opacity-40 ${!local ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-500 hover:bg-slate-200"}`}
          >
            S3 Bucket
          </button>
          <span className="text-xs text-slate-400">
            {local ? "Reads bias_clean.csv from disk" : "Pulls from configured S3 bucket"}
          </span>
        </div>

        {/* Run button */}
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
          <p className="text-sm text-red-600 flex items-center gap-2">
            <span>⚠</span> {error}
          </p>
        )}
      </div>

      {/* Active job */}
      {activeJob && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">
                Job {activeJob.job_id}
              </p>
              <p className="font-bold text-slate-900 mt-0.5">{activeJob.model_id}</p>
            </div>
            <StatusBadge status={activeJob.status} />
          </div>

          {/* Phase progress bar */}
          <PhaseBar status={activeJob.status} />

          {/* Metrics on completion */}
          {activeJob.status === "completed" && activeJob.metrics && (
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3">
                Test Metrics
              </p>
              <MetricsCard metrics={activeJob.metrics} />
            </div>
          )}

          {/* Error */}
          {activeJob.status === "failed" && activeJob.error && (
            <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {activeJob.error}
            </div>
          )}

          {/* Log terminal */}
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-2">
              Live Logs
            </p>
            <div className="bg-slate-950 rounded-xl p-4 h-64 overflow-y-auto font-mono text-xs text-slate-300 space-y-0.5">
              {activeJob.logs.map((line, i) => (
                <div
                  key={i}
                  className={
                    line.startsWith("[pipeline]")
                      ? "text-blue-400 font-semibold"
                      : line.startsWith("[cmd]")
                      ? "text-slate-500"
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
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">
            Previous Jobs
          </p>
          <div className="grid gap-3">
            {pastJobs
              .filter(j => j.job_id !== activeJob?.job_id)
              .map(job => (
                <JobCard key={job.job_id} job={job} onSelect={setActiveJob} />
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
