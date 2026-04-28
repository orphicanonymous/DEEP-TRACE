const teamMembers = [
  "Piyush Sharma",
  "Rohit Kumar",
  "Shivam Choudhary",
  "Sidarth Sharma",
  "Ujjwal Katoch",
];

const objectiveItems = [
  "Build a web-based deepfake detection system with automated preprocessing and multimodal analysis.",
  "Combine spatial, temporal, audio, and biological-signal cues for more robust audio-video verification.",
  "Generate probabilistic authenticity scores that support accountable, evidence-based review.",
];

const stackItems = [
  "React.js",
  "HTML, CSS, JavaScript",
  "Python",
  "Flask/FastAPI",
  "PyTorch/TensorFlow",
  "OpenCV",
  "Librosa",
  "MediaPipe",
];

export default function AboutPage() {
  return (
    <section className="content-shell">
      <div className="content-hero">
        <h2>DeepTrace was created to help people verify whether audio-video media is authentic or AI-generated.</h2>
        <p>
          Developed as a 2026 capstone project in the Department of Computer Science
          & Engineering (AI & ML) at Jawaharlal Nehru Government Engineering College,
          DeepTrace focuses on practical, explainable deepfake detection for high-risk
          digital media.
        </p>
      </div>

      <div className="about-layout">
        <article className="about-card about-card-wide">
          <h3>Mission</h3>
          <p>
            Generative AI has made synthetic audio and video easier to produce and
            harder to spot. DeepTrace is our effort to strengthen digital trust by
            detecting manipulated media before it can fuel misinformation, identity
            impersonation, fraud, or public harm.
          </p>
        </article>

        <article className="about-card">
          <h3>What Problem We Are Solving</h3>
          <p>
            Deepfakes now pose real risks to journalism, cybersecurity, legal review,
            and public safety. Our project addresses that challenge by treating media
            verification as a forensic task, not just a simple classification problem.
          </p>
        </article>

        <article className="about-card">
          <h3>Academic Context</h3>
          <p>
            This work was prepared as the synopsis for the capstone project
            <strong> Deep Trace: Forensic Detection of AI-Generated Audio-Video Content</strong>
            , under the supervision of <strong>Dr. Meenakshi Shruti Pal</strong>, Associate Professor
            and Head of Department.
          </p>
        </article>

        <article className="about-card">
          <h3>Core Objectives</h3>
          <ul className="about-list">
            {objectiveItems.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </article>

        <article className="about-card">
          <h3>How DeepTrace Works</h3>
          <p>
            The system analyzes uploaded media through parallel video and audio
            pipelines. It studies facial and motion patterns, inspects spectral and
            speech cues, checks lip-speech synchronization, and uses biological
            evidence such as rPPG-based heartbeat signals to estimate authenticity.
          </p>
        </article>

        <article className="about-card">
          <h3>Why This Approach Matters</h3>
          <p>
            Instead of depending on one fragile signal, DeepTrace combines spatial,
            temporal, and cross-modal evidence. That makes the final prediction more
            explainable and better suited for responsible human review.
          </p>
        </article>

        <article className="about-card">
          <h3>Project Team</h3>
          <ul className="about-list">
            {teamMembers.map((member) => (
              <li key={member}>{member}</li>
            ))}
          </ul>
        </article>

        <article className="about-card">
          <h3>Technology Stack</h3>
          <div className="about-chip-grid">
            {stackItems.map((item) => (
              <span key={item} className="about-chip">
                {item}
              </span>
            ))}
          </div>
        </article>
      </div>
    </section>
  );
}
