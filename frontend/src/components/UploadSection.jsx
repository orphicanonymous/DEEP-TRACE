import { useState } from "react";

import { supabase } from "../supabaseClient";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:5000";

function formatScoringMode(mode) {
  const labels = {
    "video+audio+fusion": "Video + Audio + Fusion",
    "video+audio": "Video + Audio",
    "video-only": "Video only",
    "audio-only": "Audio only",
    "audio-only-unreliable": "Audio only (unreliable)",
    unknown: "Unknown",
  };

  return labels[mode] || mode || "Unknown";
}

function formatTimestamp(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "Timestamp unavailable";
  }

  const totalSeconds = Math.max(0, Number(value));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = (totalSeconds % 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${seconds}`;
}

function formatMetric(value, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "Unavailable";
  }
  return `${value}${suffix}`;
}

export default function UploadSection() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError("");
    setUploadProgress(0);
    setUploadStatus("");
  };

  const uploadFile = async () => {
    if (!file || uploading) return;

    setUploading(true);
    setUploadProgress(0);
    setUploadStatus("Uploading...");
    setResult(null);
    setError("");

    // Fake progress timer during upload
    const progressTimer = setInterval(() => {
      setUploadProgress((prev) => (prev >= 40 ? prev : prev + 5));
    }, 250);

    try {
      // Step 1 — Get logged in user
      const { data: userData } = await supabase.auth.getUser();
      const user = userData.user;
      if (!user) throw new Error("You must be logged in.");

      // Step 2 — Upload file to Supabase storage
      const filePath = `${user.id}/${Date.now()}-${file.name}`;
      const { error: uploadError } = await supabase.storage
        .from("media-uploads")
        .upload(filePath, file);
      if (uploadError) throw uploadError;

      // Step 3 — Get public URL
      const { data: urlData } = supabase.storage
        .from("media-uploads")
        .getPublicUrl(filePath);

      // Step 4 — Save to database
      const { error: insertError } = await supabase.from("user_uploads").insert([
        {
          user_id: user.id,
          file_name: file.name,
          file_type: file.type,
          file_url: urlData.publicUrl,
        },
      ]);
      if (insertError) throw insertError;

      setUploadProgress(50);
      setUploadStatus("Analyzing...");

      // Fake progress during analysis
      const analysisTimer = setInterval(() => {
        setUploadProgress((prev) => (prev >= 90 ? prev : prev + 3));
      }, 400);

      // Step 5 — Send to Flask backend for analysis
      const analyzeRes = await fetch(`${BACKEND_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_url: urlData.publicUrl,
          file_name: file.name,
        }),
      });

      clearInterval(analysisTimer);

      if (!analyzeRes.ok) {
        const errData = await analyzeRes.json();
        throw new Error(errData.error || "Analysis failed");
      }

      const analysisResult = await analyzeRes.json();
      setUploadProgress(100);
      setUploadStatus("Analysis complete!");
      setResult(analysisResult);
      setFile(null);

    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong. Please try again.");
      setUploadStatus("Failed.");
    } finally {
      clearInterval(progressTimer);
      setTimeout(() => setUploading(false), 300);
    }
  };

  return (
    <div
      className="upload-card"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        setFile(e.dataTransfer.files[0]);
        setResult(null);
        setError("");
      }}
    >
      <h3>Upload Audio or Video</h3>
      <p>Drag & Drop your file here</p>
      

      {/* File input */}
      <div className="file-input-wrapper">
        <label htmlFor="fileUpload" className="custom-file-btn">
          Choose File
        </label>
        <input
          id="fileUpload"
          type="file"
          accept="audio/*,video/*"
          onChange={handleFileChange}
          hidden
          disabled={uploading}
        />
        {file && <span className="file-name">{file.name}</span>}
      </div>

      {/* Upload button */}
      <button
        className="upload-btn"
        onClick={uploadFile}
        disabled={!file || uploading}
      >
        {uploading ? uploadStatus : "Upload & Analyze"}
      </button>

      {/* Progress bar */}
      {(uploading || (uploadProgress > 0 && !result)) && (
        <div className="upload-progress-wrap">
          <div className="upload-progress-head">
            <span>{uploadStatus}</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="upload-progress-track">
            <div
              className="upload-progress-fill"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="result-error">
          ⚠️ {error}
        </div>
      )}

      {/* Result card */}
      {result && (
        <div className={`result-card ${result.verdict === "FAKE" ? "result-fake" : "result-real"}`}>

          {/* Verdict */}
          <div className="result-verdict">
            <span className="verdict-icon">
              {result.verdict === "FAKE" ? "🚨" : "✅"}
            </span>
            <span className="verdict-text">{result.verdict}</span>
          </div>

          {/* Confidence score */}
          <div className="result-score">
            <span className="score-number">{result.confidence_score}%</span>
            <span className="score-label">
              {result.verdict === "FAKE" ? "likelihood of being fake" : "likelihood of being real"}
            </span>
          </div>

          {/* Confidence label */}
          <p className="result-confidence-label">{result.confidence_label}</p>

          <div className="result-meta-grid">
            <div className="result-meta-card">
              <span className="result-meta-label">Scoring mode</span>
              <strong>{formatScoringMode(result.scoring_mode)}</strong>
            </div>
            <div className="result-meta-card">
              <span className="result-meta-label">Frames analyzed</span>
              <strong>{result.frames_analyzed ?? 0}</strong>
            </div>
          </div>

          {/* Score breakdown */}
          <div className="result-breakdown">
            {result.video_score !== null && result.video_score !== undefined && (
              <div className="breakdown-item">
                <span className="breakdown-label">🎬 Video</span>
                <div className="breakdown-bar-track">
                  <div
                    className="breakdown-bar-fill"
                    style={{ width: `${result.video_score}%` }}
                  />
                </div>
                <span className="breakdown-value">{result.video_score}%</span>
              </div>
            )}
            {result.audio_score !== null && result.audio_score !== undefined && (
              <div className="breakdown-item">
                <span className="breakdown-label">🎵 Audio</span>
                <div className="breakdown-bar-track">
                  <div
                    className="breakdown-bar-fill"
                    style={{ width: `${result.audio_score}%` }}
                  />
                </div>
                <span className="breakdown-value">{result.audio_score}%</span>
              </div>
            )}
            {result.fusion_score !== null && result.fusion_score !== undefined && (
              <div className="breakdown-item">
                <span className="breakdown-label">🔀 Fusion</span>
                <div className="breakdown-bar-track">
                  <div
                    className="breakdown-bar-fill"
                    style={{ width: `${result.fusion_score}%` }}
                  />
                </div>
                <span className="breakdown-value">{result.fusion_score}%</span>
              </div>
            )}
          </div>

          {Array.isArray(result.evidence) && result.evidence.length > 0 && (
            <div className="result-section">
              <div className="result-section-head">
                <h4>Evidence Frames</h4>
                <span>Top model signals</span>
              </div>
              <div className="evidence-list">
                {result.evidence.map((item, index) => (
                  <div className="evidence-item" key={`${item.frame_index}-${index}`}>
                    <div className="evidence-header">
                      <div>
                        <span className="evidence-kicker">Frame {item.frame_index}</span>
                        <strong>{formatTimestamp(item.timestamp_sec)}</strong>
                      </div>
                      <span className="evidence-score">{item.video_fake_score}%</span>
                    </div>
                    <p className="evidence-reason">{item.reason}</p>
                    {item.heatmap_data_url ? (
                      <div className="evidence-heatmap-wrap">
                        <img
                          className="evidence-heatmap"
                          src={item.heatmap_data_url}
                          alt={`Heatmap for evidence frame ${item.frame_index}`}
                        />
                      </div>
                    ) : null}
                    <div className="evidence-tags">
                      <span className="evidence-tag">
                        {item.used_face_crop ? "Detected face used" : "Full frame fallback"}
                      </span>
                      {item.focus_region ? (
                        <span className="evidence-tag">{item.focus_region}</span>
                      ) : null}
                      {item.cross_modal_hotspot ? (
                        <span className="evidence-tag evidence-tag-alert">Lip-sync hotspot</span>
                      ) : null}
                    </div>
                    {item.cross_modal_reason ? (
                      <p className="evidence-cross-modal-note">{item.cross_modal_reason}</p>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.forensic_signals && (
            <div className="result-section">
              <div className="result-section-head">
                <h4>Forensic Modules</h4>
                <span>Synopsis-aligned evidence</span>
              </div>
              <div className="forensic-grid">
                <div className="forensic-card">
                  <span className="forensic-label">Temporal inconsistency</span>
                  <strong>{formatMetric(result.forensic_signals.temporal_inconsistency_score, "%")}</strong>
                  <p>
                    Higher values mean stronger frame-to-frame shifts in the visual anomaly score.
                  </p>
                </div>

                <div className="forensic-card">
                  <span className="forensic-label">Lip-audio consistency</span>
                  <strong>{formatMetric(result.forensic_signals.lip_sync?.consistency_score, "%")}</strong>
                  <p>{result.forensic_signals.lip_sync?.reason || "No lip-sync report available."}</p>
                  <span className="forensic-subvalue">
                    Correlation: {formatMetric(result.forensic_signals.lip_sync?.correlation)}
                  </span>
                </div>

                <div className="forensic-card">
                  <span className="forensic-label">Biological consistency</span>
                  <strong>{formatMetric(result.forensic_signals.rppg?.consistency_score, "%")}</strong>
                  <p>{result.forensic_signals.rppg?.reason || "No rPPG report available."}</p>
                  <span className="forensic-subvalue">
                    Dominant pulse: {formatMetric(result.forensic_signals.rppg?.dominant_bpm, " bpm")}
                  </span>
                  {result.forensic_signals.rppg?.motion_artifacts && (
                    <span className="forensic-subvalue warning-text">⚠ Motion artifacts detected</span>
                  )}
                  {result.forensic_signals.rppg?.heart_rate_variability !== null && 
                   result.forensic_signals.rppg?.heart_rate_variability !== undefined && (
                    <span className="forensic-subvalue">
                      HRV score: {formatMetric(result.forensic_signals.rppg?.heart_rate_variability, "%")}
                    </span>
                  )}
                </div>
              </div>

              {Array.isArray(result.forensic_signals.lip_sync?.hotspots) &&
              result.forensic_signals.lip_sync.hotspots.length > 0 ? (
                <div className="hotspot-section">
                  <div className="result-section-head">
                    <h4>Lip-Sync Hotspots</h4>
                    <span>Localized cross-modal mismatches</span>
                  </div>
                  <div className="hotspot-list">
                    {result.forensic_signals.lip_sync.hotspots.map((hotspot, index) => (
                      <div className="hotspot-item" key={`${hotspot.timestamp_sec}-${index}`}>
                        <div className="hotspot-head">
                          <strong>{formatTimestamp(hotspot.timestamp_sec)}</strong>
                          <span className="hotspot-score">{hotspot.mismatch_score}% mismatch</span>
                        </div>
                        <p>{hotspot.reason}</p>
                        <div className="hotspot-metrics">
                          <span>Mouth motion: {formatMetric(hotspot.mouth_motion_score, "%")}</span>
                          <span>Audio energy: {formatMetric(hotspot.audio_energy_score, "%")}</span>
                          <span>{hotspot.region}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {Array.isArray(result.forensic_signals.rppg?.hotspots) &&
              result.forensic_signals.rppg.hotspots.length > 0 ? (
                <div className="hotspot-section">
                  <div className="result-section-head">
                    <h4>Pulse Signal Anomalies</h4>
                    <span>Localized regions with weak or inconsistent pulse detection</span>
                  </div>
                  <div className="hotspot-list">
                    {result.forensic_signals.rppg.hotspots.map((hotspot, index) => (
                      <div className="hotspot-item" key={`rppg-${hotspot.timestamp_sec}-${index}`}>
                        <div className="hotspot-head">
                          <strong>{formatTimestamp(hotspot.timestamp_sec)}</strong>
                          <span className="hotspot-score">{hotspot.consistency_score}% consistency</span>
                        </div>
                        <p>{hotspot.reason}</p>
                        <div className="hotspot-metrics">
                          <span>Duration: {formatMetric(hotspot.duration_sec, "s")}</span>
                          <span>Motion score: {formatMetric(hotspot.motion_score, "")}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          )}

          {Array.isArray(result.limitations) && result.limitations.length > 0 && (
            <div className="result-section">
              <div className="result-section-head">
                <h4>Review Notes</h4>
                <span>Limits and cautions</span>
              </div>
              <ul className="limitations-list">
                {result.limitations.map((item, index) => (
                  <li key={`${item}-${index}`}>{item}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Analyze another */}
          <button
            className="analyze-another-btn"
            onClick={() => {
              setResult(null);
              setUploadProgress(0);
              setUploadStatus("");
            }}
          >
            Analyze Another File
          </button>
        </div>
      )}
    </div>
  );
}
