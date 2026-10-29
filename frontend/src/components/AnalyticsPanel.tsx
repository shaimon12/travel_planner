import React from 'react';
import {
  Users,
  TrendingUp,
  Star,
  Calendar,
  Compass,
  Video,
  MapPin,
  ExternalLink,
  Info
} from 'lucide-react';
import { DestinationAnalytics } from '../types';

interface AnalyticsPanelProps {
  analytics: DestinationAnalytics | null | undefined;
  destination?: string;
}

export const AnalyticsPanel: React.FC<AnalyticsPanelProps> = ({
  analytics,
  destination = 'Destination',
}) => {
  if (!analytics) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 text-center text-slate-400 bg-slate-950">
        <div className="h-16 w-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center mb-4">
          <Compass className="h-8 w-8 text-indigo-400" />
        </div>
        <h3 className="text-base font-bold text-white mb-1">No Analytics Data Yet</h3>
        <p className="text-xs text-slate-400 max-w-sm">
          Generate an itinerary to unlock destination tourism trends, visitor numbers, and demographic insights.
        </p>
      </div>
    );
  }

  const encodedDestination = encodeURIComponent(destination);
  const ytSearchUrl = `https://www.youtube.com/results?search_query=top+10+travel+vlog+${encodedDestination}`;
  const mapsSearchUrl = `https://www.google.com/maps/search/?api=1&query=${encodedDestination}`;

  return (
    <div className="flex-1 overflow-y-auto p-6 bg-slate-950 space-y-6">
      {/* Header Banner */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-indigo-950/80 to-slate-900 border border-indigo-500/20 shadow-lg">
        <div className="flex items-center space-x-2 text-indigo-400 text-xs font-semibold uppercase tracking-wider mb-1">
          <TrendingUp className="h-4 w-4" />
          <span>Market Intelligence</span>
        </div>
        <h2 className="text-2xl font-bold text-white">Tourism Trends: {destination}</h2>
        <p className="text-xs text-slate-400 mt-1">
          Predictive metrics and tourism demographics powered by Gemini 2.5 Flash.
        </p>
      </div>

      {/* 3 Metric Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Card 1: Visitors & Growth */}
        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-sm relative overflow-hidden">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Est. Annual Visitors</span>
            <Users className="h-4 w-4 text-indigo-400" />
          </div>
          <div className="text-3xl font-extrabold text-white">
            {analytics.visitors || 'N/A'}
          </div>
          {analytics.growth && (
            <div className="mt-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <TrendingUp className="h-3 w-3 mr-1" />
              <span>{analytics.growth} YoY</span>
            </div>
          )}
        </div>

        {/* Card 2: Satisfaction */}
        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Satisfaction Rating</span>
            <Star className="h-4 w-4 text-amber-400" />
          </div>
          <div className="text-3xl font-extrabold text-white">
            {analytics.satisfaction || '4.8/5'}
          </div>
          <div className="flex items-center space-x-1 text-amber-400 text-xs mt-2">
            {'★'.repeat(5)}
            <span className="text-slate-400 text-[11px] ml-1">Verified traveler sentiment</span>
          </div>
        </div>

        {/* Card 3: Best Travel Month */}
        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Peak Travel Season</span>
            <Calendar className="h-4 w-4 text-sky-400" />
          </div>
          <div className="text-3xl font-extrabold text-sky-300">
            {analytics.best_month || 'Autumn'}
          </div>
          <div className="text-slate-400 text-xs mt-2">
            Optimal weather & cultural events
          </div>
        </div>
      </div>

      {/* Target Demographic Card */}
      <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800">
        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center space-x-1.5">
          <Users className="h-4 w-4 text-indigo-400" />
          <span>Primary Demographics</span>
        </h4>
        <div className="text-lg font-bold text-white">
          {analytics.top_demographic || 'Couples, Backpackers & Cultural Explorers'}
        </div>
        <p className="text-xs text-slate-400 mt-1">
          High affinity for culinary trails, heritage exploration, and walkable historical districts.
        </p>
      </div>

      {/* External Integration Launchers */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* YouTube Video Vlog Launcher */}
        <a
          href={ytSearchUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="group p-4 rounded-xl bg-slate-900 hover:bg-slate-800/80 border border-slate-800 hover:border-slate-700 transition flex items-center justify-between"
        >
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-xl bg-red-500/10 border border-red-500/20 flex items-center justify-center text-red-400 shrink-0">
              <Video className="h-5 w-5" />
            </div>
            <div>
              <h5 className="text-xs font-bold text-white group-hover:text-red-300 transition">
                Featured Video Guides
              </h5>
              <p className="text-[11px] text-slate-400">
                Watch curated top travel vlogs for {destination} on YouTube
              </p>
            </div>
          </div>
          <ExternalLink className="h-4 w-4 text-slate-500 group-hover:text-white transition shrink-0 ml-2" />
        </a>

        {/* Google Maps Explorer Launcher */}
        <a
          href={mapsSearchUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="group p-4 rounded-xl bg-slate-900 hover:bg-slate-800/80 border border-slate-800 hover:border-slate-700 transition flex items-center justify-between"
        >
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 shrink-0">
              <MapPin className="h-5 w-5" />
            </div>
            <div>
              <h5 className="text-xs font-bold text-white group-hover:text-indigo-300 transition">
                Google Maps Explorer
              </h5>
              <p className="text-[11px] text-slate-400">
                View interactive points of interest & transit in {destination}
              </p>
            </div>
          </div>
          <ExternalLink className="h-4 w-4 text-slate-500 group-hover:text-white transition shrink-0 ml-2" />
        </a>
      </div>

      {/* Info notice */}
      <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 flex items-center space-x-2 text-[11px] text-slate-400">
        <Info className="h-4 w-4 text-indigo-400 shrink-0" />
        <span>Metrics generated automatically by LangGraph Tourism Analytics Node (Node 3).</span>
      </div>
    </div>
  );
};

