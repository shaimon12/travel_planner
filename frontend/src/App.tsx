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
import { Loader2, Compass } from 'lucide-react';

export const App: React.FC = () => {
  const { isAuthenticated, isLoading: isAuthLoading } = useAuth();
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');

  const [trips, setTrips] = useState<TripSession[]>([]);
  const [activeTripId, setActiveTripId] = useState<string | null>(null);
  const [activeTrip, setActiveTrip] = useState<TripDetail | null>(null);
  const [activeTab, setActiveTab] = useState<'chat' | 'analytics'>('chat');
  const [isLoadingTrips, setIsLoadingTrips] = useState(false);
  const [isCreatingTrip, setIsCreatingTrip] = useState(false);
  const [isSendingMessage, setIsSendingMessage] = useState(false);

  // Load trips when authenticated
  useEffect(() => {
    if (isAuthenticated) {
      loadTrips();
    } else {
      setTrips([]);
      setActiveTripId(null);
      setActiveTrip(null);
    }
  }, [isAuthenticated]);

  // When activeTripId changes, fetch full trip details
  useEffect(() => {
    if (activeTripId && isAuthenticated) {
      loadTripDetails(activeTripId);
    } else {
      setActiveTrip(null);
    }
  }, [activeTripId, isAuthenticated]);

  const loadTrips = async () => {
    try {
      setIsLoadingTrips(true);
      const list = await apiService.fetchTrips();
      setTrips(list);
      if (list.length > 0) {
        setActiveTripId(list[0].id);
      } else {
        setActiveTripId(null);
      }
    } catch (err) {
      console.error('Failed to load trips:', err);
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

  // Unauthenticated Auth Screen (Login / Register)
  if (!isAuthenticated) {
    return (
      <div className="relative flex min-h-screen w-screen items-center justify-center bg-slate-950 p-4 overflow-hidden">
        {/* Background Ambient Glows */}
        <div className="absolute top-1/4 left-1/4 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-indigo-600/15 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-sky-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 w-full flex justify-center">
          {authMode === 'login' ? (
            <Login onSwitchToRegister={() => setAuthMode('register')} />
          ) : (
            <Register onSwitchToLogin={() => setAuthMode('login')} />
          )}
        </div>
      </div>
    );
  }

  // Authenticated Main SaaS Workspace
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 font-sans">
      {/* Persistent Sidebar */}
      <Sidebar
        trips={trips}
        activeTripId={activeTripId}
        onSelectTrip={setActiveTripId}
        onDeleteTrip={handleDeleteTrip}
        onCreateTrip={handleCreateTrip}
        isCreating={isCreatingTrip}
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
            />
          ) : (
            <AnalyticsPanel
              analytics={activeTrip?.analytics}
              destination={activeTrip?.destination}
            />
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
