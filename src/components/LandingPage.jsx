import logo from "../assets/logo.png";
import UploadSection from "./UploadSection";
import { supabase } from "../supabaseClient";

export default function LandingPage() {
  const handleLogout = async () => {
    const { error } = await supabase.auth.signOut();
    if (error) {
      console.error("Logout failed:", error.message);
    }
  };

  return (
    <>
      <nav className="navbar">
        <div className="navbar-left">
          <img src={logo} alt="logo" className="logo-img" />
          <div className="brand-text">
            <h1>DeepTrace</h1>
            <p>AI MEDIA AUTHENTICATION</p>
          </div>
        </div>

        <button className="logout-btn" onClick={handleLogout}>Logout</button>
      </nav>

      <div className="hero">
        <h2>Detect AI-Generated Audio & Video</h2>
        <p>Upload media and verify authenticity instantly.</p>
      </div>

      <UploadSection />
    </>
  );
}
