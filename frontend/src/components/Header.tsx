import React from 'react';
import {
  Compass,
  Download,
  Calendar,
  DollarSign,
  Users,
  MessageSquare,
  BarChart3,
  MapPin
} from 'lucide-react';
import { TripDetail } from '../types';

interface HeaderProps {
  trip: TripDetail | null;
  activeTab: 'chat' | 'analytics';
  onTabChange: (tab: 'chat' | 'analytics') => void;
  onDownloadItinerary: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  trip,
  activeTab,
  onTabChange,
  onDownloadItinerary,
}) => {
  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/90 backdrop-blur px-6 flex items-center justify-between z-10">
      {/* Trip Information Header */}
      <div className="flex items-center space-x-4 min-w-0">
        <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-sky-500 flex items-center justify-center shadow-lg shadow-indigo-500/20 shrink-0">
          <Compass className="h-5 w-5 text-white animate-spin-slow" />
        </div>
        
        <div className="min-w-0">
          <div className="flex items-center space-x-2">
            <h1 className="text-lg font-bold text-white truncate">
              {trip ? trip.title : 'AI Travel Planner Pro'}
            </h1>
            {trip?.destination && (
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 shrink-0">
                <MapPin className="h-3 w-3 mr-1" />
                {trip.destination}
              </span>
            )}
          </div>

          {trip && (
            <div className="flex items-center space-x-3 text-xs text-slate-400 mt-0.5 truncate">
              <span>{trip.traveler_name}</span>
              <span>•</span>
              <span className="flex items-center">
                <Calendar className="h-3 w-3 mr-1" />
                {trip.duration} Days
              </span>
              <span>•</span>
              <span className="flex items-center">
                <DollarSign className="h-3 w-3 mr-1" />
                {trip.budget}
              </span>
              <span>•</span>
              <span className="flex items-center">
                <Users className="h-3 w-3 mr-1" />
                {trip.travel_type} ({trip.group_size})
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Dual-View Tabs & Actions */}
      <div className="flex items-center space-x-3 shrink-0">
        {trip && (
          <>
            <div className="bg-slate-950 p-1 rounded-lg border border-slate-800 flex items-center space-x-1">
              <button
                onClick={() => onTabChange('chat')}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  activeTab === 'chat'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <MessageSquare className="h-3.5 w-3.5" />
                <span>Interactive Itinerary</span>
              </button>

              <button
                onClick={() => onTabChange('analytics')}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  activeTab === 'analytics'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <BarChart3 className="h-3.5 w-3.5" />
                <span>Destination Analytics</span>
              </button>
            </div>

            <button
              onClick={onDownloadItinerary}
              title="Download full Markdown itinerary"
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
            >
              <Download className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Save Markdown</span>
            </button>
          </>
        )}
      </div>
    </header>
  );
};
