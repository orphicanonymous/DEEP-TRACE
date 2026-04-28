import { useState } from "react";
import { supabase } from "../supabaseClient";
import UploadSection from "./UploadSection";
import FeaturesPage from "./FeaturesPage";
import FutureScopePage from "./FutureScopePage";
import AboutPage from "./AboutPage";

const NAV_ITEMS = [
  { id: "home", label: "Home" },
  { id: "features", label: "Features" },
  { id: "future", label: "Future Scope" },
  { id: "about", label: "About Us" },
];

function getUserDisplay(session) {
  const user = session?.user;
  const metadata = user?.user_metadata || {};
  const email = user?.email || "";
  const displayName =
    metadata.full_name ||
    metadata.name ||
    metadata.user_name ||
    (email ? email.split("@")[0] : "Signed In");

  return {
    displayName,
    email,
  };
}

export default function LandingPage({ session }) {
  const [activePage, setActivePage] = useState("home");
  const { displayName, email } = getUserDisplay(session);

  const handleLogout = async () => {
    const { error } = await supabase.auth.signOut();
    if (error) {
      console.error("Logout failed:", error.message);
    }
  };

  const renderPage = () => {
    switch (activePage) {
      case "features":
        return <FeaturesPage />;
      case "future":
        return <FutureScopePage />;
      case "about":
        return <AboutPage />;
      case "home":
      default:
        return (
          <>
            <div className="hero">
              <h2>Detect AI-Generated Audio & Video</h2>
              <p>Upload media and verify authenticity instantly.</p>
            </div>

            <UploadSection />
          </>
        );
    }
  };

  return (
    <>
      <nav className="navbar">
        <div className="navbar-left">
          <img src="/deeptrace-icon.svg" alt="DeepTrace" height="48" />
          <div className="brand-text">
            <h1>DeepTrace</h1>
            <p>AI MEDIA AUTHENTICATION</p>
          </div>
        </div>

        <div className="navbar-center">
          <div className="nav-links" role="tablist" aria-label="Site sections">
            {NAV_ITEMS.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`nav-link-btn ${activePage === item.id ? "nav-link-btn-active" : ""}`}
                onClick={() => setActivePage(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        <div className="navbar-right">
          <div className="user-chip" aria-label={`Signed in as ${displayName}`}>
            <div className="user-chip-text">
              <span className="user-chip-label">Signed in as</span>
              <span className="user-chip-name">{displayName}</span>
              {email ? <span className="user-chip-email">{email}</span> : null}
            </div>
          </div>
          <button className="logout-btn" onClick={handleLogout}>Logout</button>
        </div>
      </nav>

      {renderPage()}
    </>
  );
}
