const roadmapPlaceholders = [
  {
    title: "Browser Extension For Everyday Verification",
    text: "A future DeepTrace browser extension could run quietly while users browse platforms such as YouTube, Instagram, and news websites. It could provide instant labels like Authentic Content or Potential AI-Generated, surface confidence scores, and explain why media was flagged. With further model visualization work, the extension could also highlight suspicious regions in video frames and raise real-time alerts for manipulated content.",
  },
  {
    title: "Fake News And Newsroom Verification",
    text: "DeepTrace can evolve into a fake-news verification layer for journalists, publishers, and news platforms. In this direction, the system would help verify political speeches, viral clips, and breaking-news media before distribution. This is especially valuable during elections, public emergencies, and misinformation-heavy events where manipulated media can spread quickly.",
  },
  {
    title: "Social Media Moderation API",
    text: "A platform-facing API could help social media services such as Instagram, X, and YouTube scan uploaded media for deepfake videos, cloned voices, and suspicious multimodal content. Beyond single-file analysis, a stronger version of DeepTrace could support bulk scanning, moderation queues, and automated flagging workflows for higher-risk uploads.",
  },
  {
    title: "Digital Evidence Verification",
    text: "Another high-impact direction is legal and investigative media verification. Courts, police units, and cybercrime departments increasingly need tools to inspect fake video evidence, edited audio clips, and manipulated recordings. DeepTrace could grow into an evidence-support system that assists trained reviewers during authenticity checks while preserving human oversight.",
  },
  {
    title: "Explainability, Scoring, And Analyst Workflows",
    text: "To support these use cases well, future versions should provide richer confidence scoring, clearer reasons behind each verdict, and workflow tools such as review dashboards, case histories, media reports, and collaboration features. This would make the system more useful for analysts who need more than a binary label.",
  },
  {
    title: "Model Robustness And Deployment Readiness",
    text: "As the product expands, the model layer should continue improving in robustness, documentation, and operational maturity. Important areas include stronger evaluation across manipulation types, better audio reliability, improved deployment handling, and more accountable reporting of metrics, limitations, and failure cases.",
  },
];

export default function FutureScopePage() {
  return (
    <section className="content-shell">
      <div className="content-hero">
        <h2>The next phase is about reliability, explainability, and deployment maturity.</h2>
        <p>
          DeepTrace already has the foundation of a multimodal deepfake detection
          system. The future scope is to make it more robust, more transparent,
          and easier to use in real review environments.
        </p>
      </div>

      <div className="timeline-list">
        {roadmapPlaceholders.map((item, index) => (
          <article key={item.title} className="timeline-card">
            <span className="timeline-step">0{index + 1}</span>
            <div>
              <h3>{item.title}</h3>
              <p>{item.text}</p>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
