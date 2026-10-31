import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { ChatInterface } from './components/ChatInterface';
import { AnalyticsPanel } from './components/AnalyticsPanel';
import { Login } from './components/auth/Login';
import { Register } from './components/auth/Register';
import { apiService } from './services/api';
import { useAuth } from './context/AuthContext';
import { TripSession, TripDetail, TripCreateInput, Message } from './types';
import { Loader2, Compass, X } from 'lucide-react';

export const App: React.FC = () => {
  const { isAuthenticated, isLoading: isAuthLoading } = useAuth();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');

  const [trips, setTrips] = useState<TripSession[]>([]);
  const [activeTripId, setActiveTripId] = useState<string | null>(null);
  const [activeTrip, setActiveTrip] = useState<TripDetail | null>(null);
  const [activeTab, setActiveTab] = useState<'chat' | 'analytics'>('chat');
  const [isLoadingTrips, setIsLoadingTrips] = useState(false);
  const [isCreatingTrip, setIsCreatingTrip] = useState(false);
  const [isSendingMessage, setIsSendingMessage] = useState(false);

  // Load trips on mount / when auth state changes
  useEffect(() => {
    if (isAuthenticated) {
      loadTrips();
    } else {
      // Progressive Onboarding: Check for existing guest trip in localStorage
      const guestTripId = localStorage.getItem('guest_trip_id');
      if (guestTripId) {
        loadGuestTrip(guestTripId);
      } else {
        setTrips([]);
        setActiveTripId(null);
        setActiveTrip(null);
      }
    }
  }, [isAuthenticated]);

  // When activeTripId changes, fetch full trip details
  useEffect(() => {
    if (activeTripId) {
      loadTripDetails(activeTripId);
    } else {
      setActiveTrip(null);
    }
  }, [activeTripId]);

  const loadTrips = async () => {
    try {
      setIsLoadingTrips(true);
      const list = await apiService.fetchTrips();
      setTrips(list);
      if (list.length > 0) {
        setActiveTripId(list[0].id);
      } else {
        setActiveTripId(null);
        setActiveTrip(null);
      }
    } catch (err) {
      console.error('Failed to load trips:', err);
    } finally {
      setIsLoadingTrips(false);
    }
  };

  const loadGuestTrip = async (guestTripId: string) => {
    try {
      setIsLoadingTrips(true);
      const guestTrip = await apiService.getTrip(guestTripId);
      setTrips([guestTrip]);
      setActiveTripId(guestTrip.id);
      setActiveTrip(guestTrip);
    } catch (err) {
      console.warn('Guest trip not found or expired:', err);
      localStorage.removeItem('guest_trip_id');
      setTrips([]);
      setActiveTripId(null);
      setActiveTrip(null);
    } finally {
      setIsLoadingTrips(false);
    }
  };

  const loadTripDetails = async (id: string) => {
    try {
      const details = await apiService.getTrip(id);
      setActiveTrip(details);
    } catch (err) {
      console.error(`Failed to load trip details for ${id}:`, err);
    }
  };

  const handleCreateTrip = async (data: TripCreateInput) => {
    try {
      setIsCreatingTrip(true);
      const newTrip = await apiService.createTrip(data);
      // Store guest_trip_id for progressive onboarding if unauthenticated
      if (!isAuthenticated) {
        localStorage.setItem('guest_trip_id', newTrip.id);
      }
      setTrips(prev => [newTrip, ...prev]);
      setActiveTripId(newTrip.id);
      setActiveTrip(newTrip);
      setActiveTab('chat');
    } finally {
      setIsCreatingTrip(false);
    }
  };

  const handleDeleteTrip = async (id: string) => {
    try {
      await apiService.deleteTrip(id);
      if (localStorage.getItem('guest_trip_id') === id) {
        localStorage.removeItem('guest_trip_id');
      }
      const updatedList = trips.filter(t => t.id !== id);
      setTrips(updatedList);
      if (activeTripId === id) {
        setActiveTripId(updatedList.length > 0 ? updatedList[0].id : null);
      }
    } catch (err) {
      console.error(`Failed to delete trip ${id}:`, err);
    }
  };

  const handleSendMessage = async (content: string) => {
    if (!activeTripId || !activeTrip) return;

    // Optimistically append user message
    const tempUserMsg: Message = {
      id: `temp-${Date.now()}`,
      trip_id: activeTripId,
      role: 'user',
      content,
      created_at: new Date().toISOString(),
    };

    setActiveTrip(prev => {
      if (!prev) return null;
      return {
        ...prev,
        messages: [...prev.messages, tempUserMsg],
      };
    });

    try {
      setIsSendingMessage(true);
      const result = await apiService.sendMessage(activeTripId, content);

      setActiveTrip(prev => {
        if (!prev) return null;
        return {
          ...prev,
          messages: [...prev.messages, result.message],
          analytics: result.analytics || prev.analytics,
        };
      });
    } catch (err) {
      console.error('Error sending message to LangGraph agent:', err);
      const errorMsg: Message = {
        id: `err-${Date.now()}`,
        trip_id: activeTripId,
        role: 'assistant',
        content: '⚠️ Network or API Error. Please verify backend service and GEMINI_API_KEY configuration.',
        created_at: new Date().toISOString(),
      };
      setActiveTrip(prev => {
        if (!prev) return null;
        return {
          ...prev,
          messages: [...prev.messages, errorMsg],
        };
      });
    } finally {
      setIsSendingMessage(false);
    }
  };

  const handleDownloadItinerary = () => {
    if (!activeTrip || activeTrip.messages.length === 0) return;

    const assistantMessages = activeTrip.messages.filter(m => m.role === 'assistant');
    const content =
      assistantMessages.length > 0
        ? assistantMessages[0].content
        : `# ${activeTrip.title}\nDestination: ${activeTrip.destination}`;

    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const safeDest = (activeTrip.destination || 'My_Trip').replace(/\s+/g, '_');
    link.href = url;
    link.download = `Itinerary_${safeDest}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const openAuth = (mode: 'login' | 'register' = 'login') => {
    setAuthMode(mode);
    setShowAuthModal(true);
  };

  // Splash Screen during auth initialization
  if (isAuthLoading) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-slate-950 text-slate-100">
        <div className="flex flex-col items-center space-y-4">
          <div className="h-14 w-14 rounded-2xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center shadow-lg shadow-indigo-600/30">
            <Compass className="h-8 w-8 text-indigo-400 animate-spin" />
          </div>
          <div className="flex items-center space-x-2 text-xs text-slate-400">
            <Loader2 className="h-4 w-4 animate-spin text-indigo-400" />
            <span>Validating authentication session...</span>
          </div>
        </div>
      </div>
    );
  }

  // Progressive Onboarding: Unauthenticated users have direct access to the main workspace!
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 font-sans relative">
      {/* Persistent Sidebar */}
      <Sidebar
        trips={trips}
        activeTripId={activeTripId}
        onSelectTrip={setActiveTripId}
        onDeleteTrip={handleDeleteTrip}
        onCreateTrip={handleCreateTrip}
        isCreating={isCreatingTrip}
        onOpenAuth={openAuth}
      />

      {/* Main Dual-View Workspace */}
      <div className="flex-1 flex flex-col h-full min-w-0 overflow-hidden">
        <Header
          trip={activeTrip}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          onDownloadItinerary={handleDownloadItinerary}
        />

        <main className="flex-1 flex overflow-hidden">
          {isLoadingTrips ? (
            <div className="flex-1 flex items-center justify-center space-x-2 text-slate-400">
              <Loader2 className="h-6 w-6 animate-spin text-indigo-400" />
              <span className="text-sm font-medium">Loading your trip itineraries...</span>
            </div>
          ) : activeTab === 'chat' ? (
            <ChatInterface
              messages={activeTrip?.messages || []}
              onSendMessage={handleSendMessage}
              isSending={isSendingMessage}
              destination={activeTrip?.destination}
              onRequireAuth={openAuth}
            />
          ) : (
            <AnalyticsPanel
              analytics={activeTrip?.analytics}
              destination={activeTrip?.destination}
            />
          )}
        </main>
      </div>

      {/* Auth Modal Overlay (Login / Register) */}
      {showAuthModal && !isAuthenticated && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fade-in">
          <div className="relative w-full max-w-md">
            <button
              onClick={() => setShowAuthModal(false)}
              className="absolute -top-3 -right-3 z-20 h-8 w-8 rounded-full bg-slate-800 border border-slate-700 text-slate-300 hover:text-white flex items-center justify-center transition shadow-lg"
              title="Close"
            >
              <X className="h-4 w-4" />
            </button>
            {authMode === 'login' ? (
              <Login onSwitchToRegister={() => setAuthMode('register')} />
            ) : (
              <Register onSwitchToLogin={() => setAuthMode('login')} />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
