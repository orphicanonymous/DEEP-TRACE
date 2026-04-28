import { useEffect, useState } from "react";
import { supabase } from "./supabaseClient";
import LoginModal from "./components/LoginModal";
import LandingPage from "./components/LandingPage";
import SplashCursor from "./components/SplashCursor";

function App() {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const getSession = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      setSession(session);
      setLoading(false);
    };

    getSession();

    const { data: listener } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setSession(session);
      }
    );

    return () => {
      listener.subscription.unsubscribe();
    };
  }, []);

  if (loading) return null;

  return (
    <>
      <SplashCursor />
      <div className="app-content">
        {session ? <LandingPage session={session} /> : <LoginModal />}
      </div>
    </>
  );
}

export default App;
