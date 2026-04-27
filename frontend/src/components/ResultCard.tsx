interface PredictResponse {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  model_used: string;
}

interface ResultCardProps {
  result: PredictResponse;
}

const LABEL_CONFIG: Record<string, { bg: string; text: string; border: string; bar: string; icon: string }> = {
  left: {
    bg: "bg-blue-50",
    text: "text-blue-700",
    border: "border-blue-200",
    bar: "bg-blue-500",
    icon: "←",
  },
  center: {
    bg: "bg-emerald-50",
    text: "text-emerald-700",
    border: "border-emerald-200",
    bar: "bg-emerald-500",
    icon: "◎",
  },
  right: {
    bg: "bg-red-50",
    text: "text-red-700",
    border: "border-red-200",
    bar: "bg-red-500",
    icon: "→",
  },
};

const LABEL_DESCRIPTIONS: Record<string, string> = {
  left: "This article leans toward left-leaning political perspective.",
  center: "This article appears balanced or centrist in its coverage.",
  right: "This article leans toward right-leaning political perspective.",
};

const DEFAULT_CONFIG = {
  bg: "bg-slate-50",
  text: "text-slate-700",
  border: "border-slate-200",
  bar: "bg-slate-500",
  icon: "?",
};

export default function ResultCard({ result }: ResultCardProps) {
  const label = result.label.toLowerCase();
  const cfg = LABEL_CONFIG[label] ?? DEFAULT_CONFIG;
  const description = LABEL_DESCRIPTIONS[label] ?? "";
  const confidencePct = Math.round(result.confidence * 100);

  const sortedProbs = Object.entries(result.probabilities).sort(([a], [b]) =>
    a.localeCompare(b)
  );

  return (
    <div className={`rounded-2xl border ${cfg.border} ${cfg.bg} p-6 space-y-6 animate-fade-in`}>
      {/* Primary label */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-1">
            Detected Bias
          </p>
          <div className="flex items-center gap-3">
            <span className={`text-4xl font-bold capitalize ${cfg.text}`}>
              {cfg.icon} {result.label}
            </span>
          </div>
          <p className="text-sm text-slate-500 mt-1">{description}</p>
        </div>
        <div className="text-right">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-1">
            Confidence
          </p>
          <p className={`text-4xl font-bold ${cfg.text}`}>{confidencePct}%</p>
        </div>
      </div>

      {/* Confidence bar */}
      <div>
        <div className="flex justify-between text-xs text-slate-400 mb-1">
          <span>Confidence</span>
          <span>{confidencePct}%</span>
        </div>
        <div className="w-full bg-slate-200 rounded-full h-2">
          <div
            className={`h-2 rounded-full ${cfg.bar} transition-width`}
            style={{ width: `${confidencePct}%` }}
          />
        </div>
      </div>

      {/* All class probabilities */}
      <div>
        <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3">
          All Probabilities
        </p>
        <div className="space-y-3">
          {sortedProbs.map(([cls, prob]) => {
            const clsCfg = LABEL_CONFIG[cls.toLowerCase()] ?? DEFAULT_CONFIG;
            const pct = Math.round(prob * 100);
            const isWinner = cls.toLowerCase() === label;
            return (
              <div key={cls}>
                <div className="flex justify-between text-sm mb-1">
                  <span className={`font-medium capitalize ${isWinner ? clsCfg.text : "text-slate-500"}`}>
                    {isWinner ? "★ " : ""}{cls}
                  </span>
                  <span className={`font-medium ${isWinner ? clsCfg.text : "text-slate-400"}`}>
                    {pct}%
                  </span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-width ${clsCfg.bar} ${!isWinner ? "opacity-40" : ""}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer metadata */}
      <div className="pt-2 border-t border-slate-200 flex items-center gap-2">
        <span className="inline-flex items-center rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-medium text-slate-600">
          {result.model_used === "bert" ? "BERT" : "DistilBERT"}
        </span>
        <span className="text-xs text-slate-400">fine-tuned for political bias classification</span>
      </div>
    </div>
  );
}
