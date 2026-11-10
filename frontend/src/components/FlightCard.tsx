import React from 'react';
import { Plane, Clock, ExternalLink } from 'lucide-react';
import { FlightOption } from '../types';

interface FlightCardProps {
  flight: FlightOption;
}

export const FlightCard: React.FC<FlightCardProps> = ({ flight }) => {
  const bookingUrl = flight.link || flight.booking_url || '#';

  return (
    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800/90 hover:border-indigo-400 dark:hover:border-indigo-500/40 rounded-xl p-4 shadow-sm hover:shadow-indigo-500/10 transition duration-200 flex flex-col justify-between space-y-3 group">
      {/* Top Header: Airline & Tag */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="h-7 w-7 rounded-lg bg-sky-50 dark:bg-sky-500/10 border border-sky-200 dark:border-sky-500/20 flex items-center justify-center text-sky-600 dark:text-sky-400 shrink-0">
            <Plane className="h-4 w-4 transform group-hover:translate-x-0.5 transition-transform" />
          </div>
          <div>
            <h4 className="text-xs font-bold text-slate-900 dark:text-white leading-tight">{flight.airline}</h4>
            {flight.flight_number && (
              <span className="text-[10px] text-slate-500 dark:text-slate-400 font-mono">{flight.flight_number}</span>
            )}
          </div>
        </div>

        {flight.tag && (
          <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200 dark:bg-emerald-500/15 dark:text-emerald-300 dark:border-emerald-500/30">
            {flight.tag}
          </span>
        )}
      </div>

      {/* Flight Timing & Stops Route */}
      <div className="bg-slate-50 dark:bg-slate-950/60 rounded-lg p-2.5 border border-slate-200 dark:border-slate-800/70 flex items-center justify-between text-xs">
        {/* Departure */}
        <div className="text-left">
          <span className="block text-[10px] text-slate-500 font-medium uppercase tracking-wider">Departs</span>
          <span className="text-sm font-extrabold text-slate-900 dark:text-white">{flight.departure}</span>
        </div>

        {/* Center Route Line & Duration */}
        <div className="flex flex-col items-center px-3">
          <span className="text-[10px] text-slate-500 dark:text-slate-400 flex items-center mb-0.5">
            <Clock className="h-2.5 w-2.5 mr-1 text-slate-400 dark:text-slate-500" />
            {flight.duration || '9h 15m'}
          </span>
          <div className="flex items-center space-x-1.5 w-16">
            <div className="h-[1px] flex-1 bg-slate-300 dark:bg-slate-700" />
            <Plane className="h-2.5 w-2.5 text-indigo-500 dark:text-indigo-400 transform rotate-90" />
            <div className="h-[1px] flex-1 bg-slate-300 dark:bg-slate-700" />
          </div>
          <span className="text-[9px] text-indigo-600 dark:text-indigo-300 font-medium mt-0.5">
            {flight.stops || 'Nonstop'}
          </span>
        </div>

        {/* Arrival */}
        <div className="text-right">
          <span className="block text-[10px] text-slate-500 font-medium uppercase tracking-wider">Arrives</span>
          <span className="text-sm font-extrabold text-slate-900 dark:text-white">{flight.arrival}</span>
        </div>
      </div>

      {/* Bottom Footer: Price & CTA */}
      <div className="flex items-center justify-between pt-1 border-t border-slate-200 dark:border-slate-800/60">
        <div>
          <span className="block text-[10px] text-slate-500">Est. Total</span>
          <div className="flex items-baseline space-x-1">
            <span className="text-base font-black text-emerald-600 dark:text-emerald-400">{flight.price}</span>
            <span className="text-[10px] text-slate-500 dark:text-slate-400">/ traveler</span>
          </div>
        </div>

        {/*
          PRODUCTION TRAVEL AFFILIATE INTEGRATION PLACEHOLDER:
          Append your affiliate tracking ID (e.g., Travelpayouts marker or Skyscanner campaign tag)
          Example: const affiliateUrl = `${bookingUrl}&marker=${AFFILIATE_MARKER}&subid=travel_planner_pro`;
        */}
        <a
          href={bookingUrl}
          target="_blank"
          rel="noopener noreferrer"
          data-affiliate-partner="travelpayouts"
          data-affiliate-tracking-id="AFFILIATE_TRACKING_ID_PLACEHOLDER"
          data-affiliate-campaign="flight_booking_cta"
          className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 active:scale-95 text-white font-semibold text-xs shadow-md shadow-indigo-600/25 transition"
        >
          <span>Book Ticket ✈️</span>
          <ExternalLink className="h-3 w-3 opacity-80" />
        </a>
      </div>
    </div>
  );
};
