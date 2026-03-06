import { supabase } from "../supabaseClient";
import { useState } from "react";

export default function UploadSection() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState("");

  const uploadFile = async () => {
    if (!file || uploading) return;

    setUploading(true);
    setUploadProgress(0);
    setUploadStatus("Uploading...");

    // Supabase upload does not expose granular progress in this API path,
    // so we show a smooth loading percentage until the request completes.
    const progressTimer = setInterval(() => {
      setUploadProgress((prev) => (prev >= 90 ? prev : prev + 5));
    }, 250);

    try {
      const { data: userData } = await supabase.auth.getUser();
      const user = userData.user;

      if (!user) {
        throw new Error("You must be logged in to upload files.");
      }

      const filePath = `${user.id}/${Date.now()}-${file.name}`;

      const { error: uploadError } = await supabase.storage
        .from("media-uploads")
        .upload(filePath, file);

      if (uploadError) {
        throw uploadError;
      }

      const { data: urlData } = supabase.storage
        .from("media-uploads")
        .getPublicUrl(filePath);

      const { error: insertError } = await supabase.from("user_uploads").insert([
        {
          user_id: user.id,
          file_name: file.name,
          file_type: file.type,
          file_url: urlData.publicUrl,
        },
      ]);

      if (insertError) {
        throw insertError;
      }

      setUploadProgress(100);
      setUploadStatus("Upload complete");
      setFile(null);
    } catch (error) {
      console.error(error);
      setUploadStatus("Upload failed. Please try again.");
    } finally {
      clearInterval(progressTimer);
      setTimeout(() => {
        setUploading(false);
      }, 300);
    }
  };

  return (
    <div
      className="upload-card"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        setFile(e.dataTransfer.files[0]);
      }}
    >
      <h3>Upload Audio or Video</h3>
      <p>Drag & Drop your file here</p>

      <div className="file-input-wrapper">
        <label htmlFor="fileUpload" className="custom-file-btn">
          Choose File
        </label>

        <input
          id="fileUpload"
          type="file"
          accept="audio/*,video/*"
          onChange={(e) => setFile(e.target.files[0])}
          hidden
          disabled={uploading}
        />

        {file && <span className="file-name">{file.name}</span>}
      </div>

      <button
        className="upload-btn"
        onClick={uploadFile}
        disabled={!file || uploading}
      >
        {uploading ? "Uploading..." : "Upload & Analyze"}
      </button>

      {(uploading || uploadProgress > 0) && (
        <div className="upload-progress-wrap">
          <div className="upload-progress-head">
            <span>{uploadStatus || "Uploading..."}</span>
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
    </div>
  );
}
