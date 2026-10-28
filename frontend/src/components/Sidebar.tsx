import React, { useState } from 'react';
import {
  Plus,
  Trash2,
  Compass,
  Sparkles,
  ChevronRight,
  ExternalLink,
  Loader2,
  MapPin
} from 'lucide-react';
import { TripSession, TripCreateInput } from '../types';

interface SidebarProps {
  trips: TripSession[];
  activeTripId: string | null;
  onSelectTrip: (tripId: string) => void;
  onDeleteTrip: (tripId: string) => void;
  onCreateTrip: (data: TripCreateInput) => Promise<void>;
  isCreating: boolean;
}

const INTEREST_OPTIONS = [
  'History 🏛️',
  'Nature 🌳',
  'Food 🍜',
  'Adventure 🧗',
  'Shopping 🛍️',
  'Art 🎨',
  'Nightlife 🍹',
];

const DIETARY_OPTIONS = [
  'Vegetarian',
  'Vegan',
  'Halal',
  'Gluten-Free',
  'Kosher',
  'Seafood Allergy',
];

export const Sidebar: React.FC<SidebarProps> = ({
  trips,
  activeTripId,
  onSelectTrip,
  onDeleteTrip,
  onCreateTrip,
  isCreating,
}) => {
  const [showForm, setShowForm] = useState(false);

  // Form State
  const [travelerName, setTravelerName] = useState('Alex');
  const [rawDestination, setRawDestination] = useState('');
  const [startDate, setStartDate] = useState(new Date().toISOString().split('T')[0]);
  const [duration, setDuration] = useState(3);
  const [budget, setBudget] = useState('Moderate');
  const [travelType, setTravelType] = useState('Solo');
  const [groupSize, setGroupSize] = useState(1);
  const [interests, setInterests] = useState<string[]>(['History 🏛️', 'Food 🍜']);
  const [dietary, setDietary] = useState<string[]>([]);
  const [formError, setFormError] = useState<string | null>(null);

  const toggleInterest = (interest: string) => {
    setInterests(prev =>
      prev.includes(interest) ? prev.filter(i => i !== interest) : [...prev, interest]
    );
  };

  const toggleDietary = (item: string) => {
    setDietary(prev =>
      prev.includes(item) ? prev.filter(d => d !== item) : [...prev, item]
    );
  };

  const handleCreateSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);

    if (!travelerName.trim() || !rawDestination.trim()) {
      setFormError('Please provide both traveler name and destination.');
      return;
    }

    try {
      await onCreateTrip({
        traveler_name: travelerName.trim(),
        raw_destination: rawDestination.trim(),
        start_date: startDate,
        duration: Number(duration),
        budget,
        travel_type: travelType,
        group_size: Number(groupSize),
        interests,
        dietary_restrictions: dietary,
      });
      setShowForm(false);
      setRawDestination('');
    } catch (err: any) {
      setFormError(err.response?.data?.detail || 'Failed to initialize trip. Please check connection.');
    }
  };

  return (
    <aside className="w-80 md:w-96 bg-slate-900 border-r border-slate-800 flex flex-col h-full shrink-0 select-none">
      {/* Brand & New Trip Action */}
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="h-8 w-8 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-white shadow">
            🌍
          </div>
          <div>
            <h2 className="text-sm font-bold text-white leading-tight">AI Travel Planner</h2>
            <p className="text-[11px] text-slate-400">Enterprise Multi-Agent Platform</p>
          </div>
        </div>

        <button
          onClick={() => setShowForm(!showForm)}
          className={`flex items-center space-x-1 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
            showForm
              ? 'bg-slate-800 text-slate-300'
              : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-md shadow-indigo-600/30'
          }`}
        >
          {showForm ? 'Cancel' : <><Plus className="h-3.5 w-3.5 mr-0.5" /> New Trip</>}
        </button>
      </div>

      {/* Main Content: Form or History List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {showForm ? (
          /* New Trip Form */
          <form onSubmit={handleCreateSubmit} className="space-y-4 text-xs bg-slate-950/70 p-4 rounded-xl border border-slate-800/80">
            <div className="flex items-center space-x-2 text-indigo-400 font-semibold mb-2">
              <Sparkles className="h-4 w-4" />
              <span>Configure Your Trip</span>
            </div>

            {formError && (
              <div className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-[11px]">
                {formError}
              </div>
            )}

            <div>
              <label className="block text-slate-300 font-medium mb-1">Traveler Name</label>
              <input
                type="text"
                value={travelerName}
                onChange={e => setTravelerName(e.target.value)}
                placeholder="e.g. Alex"
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
                required
              />
            </div>

            <div>
              <label className="block text-slate-300 font-medium mb-1">Destination</label>
              <div className="relative">
                <MapPin className="h-3.5 w-3.5 absolute left-3 top-2.5 text-slate-500" />
                <input
                  type="text"
                  value={rawDestination}
                  onChange={e => setRawDestination(e.target.value)}
                  placeholder="e.g. Kyoto, Japan or Paris"
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-8 pr-3 py-1.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-slate-300 font-medium mb-1">Start Date</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={e => setStartDate(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-100 focus:outline-none focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="block text-slate-300 font-medium mb-1">Duration (Days)</label>
                <input
                  type="number"
                  min="1"
                  max="30"
                  value={duration}
                  onChange={e => setDuration(Number(e.target.value))}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-100 focus:outline-none focus:border-indigo-500"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-slate-300 font-medium mb-1">Budget</label>
                <select
                  value={budget}
                  onChange={e => setBudget(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-slate-100 focus:outline-none focus:border-indigo-500"
                >
                  <option value="Shoestring">Shoestring</option>
                  <option value="Moderate">Moderate</option>
                  <option value="Luxury">Luxury</option>
                </select>
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Style</label>
                <select
                  value={travelType}
                  onChange={e => setTravelType(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-slate-100 focus:outline-none focus:border-indigo-500"
                >
                  <option value="Solo">Solo</option>
                  <option value="Couple">Couple</option>
                  <option value="Family">Family</option>
                  <option value="Friends">Friends</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-slate-300 font-medium mb-1">Total Travelers</label>
              <input
                type="number"
                min="1"
                max="20"
                value={groupSize}
                onChange={e => setGroupSize(Number(e.target.value))}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-100 focus:outline-none focus:border-indigo-500"
              />
            </div>

            {/* Interests Chips */}
            <div>
              <label className="block text-slate-300 font-medium mb-1.5">Core Interests</label>
              <div className="flex flex-wrap gap-1.5">
                {INTEREST_OPTIONS.map(interest => {
                  const selected = interests.includes(interest);
                  return (
                    <button
                      type="button"
                      key={interest}
                      onClick={() => toggleInterest(interest)}
                      className={`px-2 py-1 rounded-md text-[11px] font-medium transition ${
                        selected
                          ? 'bg-indigo-600 text-white'
                          : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                      }`}
                    >
                      {interest}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Dietary Chips */}
            <div>
              <label className="block text-slate-300 font-medium mb-1.5">Dietary Restrictions</label>
              <div className="flex flex-wrap gap-1.5">
                {DIETARY_OPTIONS.map(d => {
                  const selected = dietary.includes(d);
                  return (
                    <button
                      type="button"
                      key={d}
                      onClick={() => toggleDietary(d)}
                      className={`px-2 py-1 rounded-md text-[11px] font-medium transition ${
                        selected
                          ? 'bg-emerald-600 text-white'
                          : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                      }`}
                    >
                      {d}
                    </button>
                  );
                })}
              </div>
            </div>

            <button
              type="submit"
              disabled={isCreating}
              className="w-full mt-2 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 font-bold text-white shadow-lg shadow-indigo-600/30 flex items-center justify-center space-x-2 transition disabled:opacity-50"
            >
              {isCreating ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Orchestrating Agents...</span>
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
                  <span>Generate Itinerary</span>
                </>
              )}
            </button>
          </form>
        ) : (
          /* Trips History List */
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs font-semibold text-slate-400 px-1 mb-1">
              <span>Trip Sessions ({trips.length})</span>
            </div>

            {trips.length === 0 ? (
              <div className="text-center py-8 px-4 bg-slate-950/40 rounded-xl border border-dashed border-slate-800 text-slate-400 text-xs">
                <Compass className="h-8 w-8 mx-auto text-slate-600 mb-2" />
                <p className="font-medium text-slate-300">No active trips yet</p>
                <p className="text-[11px] text-slate-500 mt-1">Click "New Trip" to launch the AI concierge</p>
              </div>
            ) : (
              trips.map(trip => {
                const isActive = trip.id === activeTripId;
                return (
                  <div
                    key={trip.id}
                    onClick={() => onSelectTrip(trip.id)}
                    className={`group relative flex items-center justify-between p-3 rounded-xl cursor-pointer border transition-all ${
                      isActive
                        ? 'bg-indigo-600/15 border-indigo-500/40 text-white shadow-sm'
                        : 'bg-slate-950/50 border-slate-800/80 text-slate-300 hover:bg-slate-800/60 hover:border-slate-700'
                    }`}
                  >
                    <div className="min-w-0 flex-1 pr-2">
                      <div className="flex items-center space-x-1.5 font-semibold text-xs truncate">
                        <MapPin className="h-3.5 w-3.5 text-indigo-400 shrink-0" />
                        <span className="truncate">{trip.title}</span>
                      </div>
                      <div className="flex items-center space-x-2 text-[11px] text-slate-400 mt-1">
                        <span>{trip.duration}d</span>
                        <span>•</span>
                        <span>{trip.budget}</span>
                        <span>•</span>
                        <span>{trip.traveler_name}</span>
                      </div>
                    </div>

                    <div className="flex items-center space-x-1">
                      <button
                        onClick={e => {
                          e.stopPropagation();
                          if (confirm(`Delete ${trip.title}?`)) {
                            onDeleteTrip(trip.id);
                          }
                        }}
                        title="Delete trip session"
                        className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-rose-500/20 text-slate-400 hover:text-rose-400 rounded-lg transition"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                      <ChevronRight className={`h-4 w-4 ${isActive ? 'text-indigo-400' : 'text-slate-600'}`} />
                    </div>
                  </div>
                );
              })
            )}
          </div>
        )}
      </div>

      {/* Developer Footer */}
      <div className="p-3 border-t border-slate-800 bg-slate-950/60 text-[11px] text-slate-400 flex items-center justify-between">
        <div>
          <span className="text-slate-300 font-medium">Shaimon Rahman</span>
          <span className="block text-[10px] text-slate-500">Gemini 2.5 Flash & LangGraph</span>
        </div>
        <a
          href="https://www.linkedin.com/in/shaimonrahman"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center space-x-1 px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-sky-400 hover:text-sky-300 text-[10px] transition"
        >
          <span>LinkedIn</span>
          <ExternalLink className="h-3 w-3" />
        </a>
      </div>
    </aside>
  );
};
