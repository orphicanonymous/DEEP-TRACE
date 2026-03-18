import { supabase } from "../supabaseClient";
import { useState } from "react";

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
      const analyzeRes = await fetch("http://127.0.0.1:5000/analyze", {
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
      {(uploading || uploadProgress > 0) && (
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
