const featurePlaceholders = [
  {
    title: "Multimodal Detection",
    text: "DeepTrace combines three analysis paths: an Xception-based video detector, a wav2vec2-based audio detector, and a fusion transformer that merges both embeddings when the audio branch is reliable.",
  },
  {
    title: "Frame And Audio Processing",
    text: "For video, the system samples 20 frames, detects faces with MTCNN, and falls back to the full frame when no face is found. For audio, it converts to mono, resamples to 16 kHz, trims or pads to 3 seconds, and normalizes before inference.",
  },
  {
    title: "Decision Support With Score Breakdown",
    text: "The platform returns a final verdict together with confidence score, confidence label, and branch-level outputs such as video score, audio score, and fusion score. This makes the result easier to inspect instead of presenting only a single opaque label.",
  },
  {
    title: "Reliability Guardrails",
    text: "DeepTrace includes an audio reliability gate. If the audio score stays too close to chance, that branch is excluded from fusion and the system falls back to a safer scoring mode such as video-only analysis.",
  },
  {
    title: "Adaptive Scoring Logic",
    text: "When all reliable branches are available, the final score is weighted across video, audio, and fusion. If one branch fails or is unreliable, the backend automatically shifts to the best available evidence instead of forcing a broken output.",
  },
  {
    title: "Built For Human Review",
    text: "DeepTrace is intended as a review aid rather than a final forensic authority. Low-confidence outputs are explicitly flagged for manual review, making the tool more suitable for accountable verification workflows.",
  },
];

export default function FeaturesPage() {
  return (
    <section className="content-shell">
      <div className="content-hero">
        <h2>DeepTrace is designed as a practical multimodal verification pipeline.</h2>
        <p>
          It does not rely on a single signal. Instead, it combines visual evidence,
          audio evidence, and cross-modal reasoning to estimate whether uploaded media
          is likely real or manipulated.
        </p>
      </div>

      <div className="info-grid">
        {featurePlaceholders.map((item) => (
          <article key={item.title} className="info-card">
            <h3>{item.title}</h3>
            <p>{item.text}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
