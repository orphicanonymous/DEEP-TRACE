import { supabase } from "../supabaseClient";
import logo from "../assets/logo.png";

export default function LoginModal() {

  const handleGoogleLogin = async () => {
    await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: window.location.origin,
      },
    });
  };

  return (
    <div className="login-page">
     
      <div className="login-card">
        <img src={logo} className="login-logo fade-in" alt="DeepTrace Logo" />
        <h1 className="brand-title fade-in">DeepTrace</h1>
        <p className="brand-subtitle fade-in-delay">AI Media Authenticator</p>

        <button className="google-button fade-in-delay" onClick={handleGoogleLogin}>
          Continue with Google
        </button>
      </div>
    </div>
  );
}
