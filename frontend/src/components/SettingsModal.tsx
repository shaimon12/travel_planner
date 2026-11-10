import React from 'react';
import { X, Settings, Key, Cpu, Globe, Shield, Sparkles } from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fade-in">
      <div className="w-full max-w-lg bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl relative text-left space-y-5">
        {/* Modal Header */}
        <div className="flex items-start justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-xl bg-indigo-600/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center shrink-0">
              <Settings className="h-5 w-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white flex items-center space-x-2">
                <span>Advanced Engine Settings</span>
                <span className="text-[10px] font-mono font-normal uppercase bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 px-1.5 py-0.5 rounded">
                  BYOK
                </span>
              </h3>
              <p className="text-xs text-slate-400">
                Custom LLM endpoints and developer provider keys.
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 transition p-1 rounded-lg hover:bg-slate-800"
            title="Close modal"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* 99% User Notice Banner */}
        <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-xs flex items-start space-x-2.5">
          <Shield className="h-4 w-4 shrink-0 mt-0.5 text-emerald-400" />
          <div>
            <span className="font-semibold block text-emerald-200">Zero-Config Free Tier Active</span>
            <span className="text-emerald-300/80 text-[11px] leading-relaxed">
              99% of travelers don't need to change anything here. Fast PostgreSQL semantic caching and OpenRouter auto-routing are fully active.
            </span>
          </div>
        </div>

        {/* Skeleton Form Fields for BYOK */}
        <div className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5 flex items-center justify-between">
              <span className="flex items-center space-x-1.5">
                <Key className="h-3.5 w-3.5 text-slate-400" />
                <span>Custom API Key (BYOK)</span>
              </span>
              <span className="text-[10px] text-slate-500">Optional</span>
            </label>
            <input
              type="password"
              placeholder="sk-or-v1-..."
              disabled
              className="w-full bg-slate-950/70 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-500 placeholder-slate-600 cursor-not-allowed"
            />
            <span className="text-[10px] text-slate-500 mt-1 block">
              Direct OpenRouter or OpenAI keys will bypass shared rate limits in upcoming BYOK update.
            </span>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5 flex items-center justify-between">
              <span className="flex items-center space-x-1.5">
                <Cpu className="h-3.5 w-3.5 text-slate-400" />
                <span>Primary Model Identifier</span>
              </span>
              <span className="text-[10px] text-slate-500">Default: openrouter/free</span>
            </label>
            <input
              type="text"
              defaultValue="openrouter/free"
              disabled
              className="w-full bg-slate-950/70 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-500 placeholder-slate-600 cursor-not-allowed font-mono"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5 flex items-center justify-between">
              <span className="flex items-center space-x-1.5">
                <Globe className="h-3.5 w-3.5 text-slate-400" />
                <span>API Base URL</span>
              </span>
              <span className="text-[10px] text-slate-500">OpenAI compatible</span>
            </label>
            <input
              type="text"
              defaultValue="https://openrouter.ai/api/v1"
              disabled
              className="w-full bg-slate-950/70 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-500 placeholder-slate-600 cursor-not-allowed font-mono"
            />
          </div>
        </div>

        {/* Modal Footer */}
        <div className="pt-2 border-t border-slate-800 flex items-center justify-between">
          <span className="text-[10px] text-slate-500 flex items-center">
            <Sparkles className="h-3 w-3 mr-1 text-indigo-400" />
            Semantic Cache & Auto-Router Enabled
          </span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
