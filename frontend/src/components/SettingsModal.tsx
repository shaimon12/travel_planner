import React, { useState, useEffect } from 'react';
import { X, Settings, Key, Cpu, Globe, Shield, Sparkles, Check, RotateCcw, Zap } from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose }) => {
  const [customApiKey, setCustomApiKey] = useState('');
  const [customModel, setCustomModel] = useState('');
  const [customBaseUrl, setCustomBaseUrl] = useState('');
  const [saved, setSaved] = useState(false);

  // Load stored settings on mount or modal open
  useEffect(() => {
    if (isOpen) {
      setCustomApiKey(localStorage.getItem('customApiKey') || '');
      setCustomModel(localStorage.getItem('customModel') || '');
      setCustomBaseUrl(localStorage.getItem('customBaseUrl') || '');
      setSaved(false);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const hasActiveByok = Boolean(localStorage.getItem('customApiKey')?.trim());

  const handleSave = () => {
    const trimmedKey = customApiKey.trim();
    const trimmedModel = customModel.trim();
    const trimmedBaseUrl = customBaseUrl.trim();

    if (trimmedKey) {
      localStorage.setItem('customApiKey', trimmedKey);
    } else {
      localStorage.removeItem('customApiKey');
    }

    if (trimmedModel) {
      localStorage.setItem('customModel', trimmedModel);
    } else {
      localStorage.removeItem('customModel');
    }

    if (trimmedBaseUrl) {
      localStorage.setItem('customBaseUrl', trimmedBaseUrl);
    } else {
      localStorage.removeItem('customBaseUrl');
    }

    setSaved(true);
    setTimeout(() => {
      setSaved(false);
    }, 2000);
  };

  const handleReset = () => {
    localStorage.removeItem('customApiKey');
    localStorage.removeItem('customModel');
    localStorage.removeItem('customBaseUrl');
    setCustomApiKey('');
    setCustomModel('');
    setCustomBaseUrl('');
    setSaved(false);
  };

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
                Configure your own OpenAI-compatible API key and endpoint.
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

        {/* Status Banner */}
        {hasActiveByok ? (
          <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 text-xs flex items-start space-x-2.5">
            <Zap className="h-4 w-4 shrink-0 mt-0.5 text-indigo-400" />
            <div>
              <span className="font-semibold block text-indigo-200">BYOK Override Active</span>
              <span className="text-indigo-300/80 text-[11px] leading-relaxed">
                Requests inject your custom credentials via headers. Outbound calls bypass shared rate limits and use your private client key.
              </span>
            </div>
          </div>
        ) : (
          <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-xs flex items-start space-x-2.5">
            <Shield className="h-4 w-4 shrink-0 mt-0.5 text-emerald-400" />
            <div>
              <span className="font-semibold block text-emerald-200">Zero-Config Free Tier Active</span>
              <span className="text-emerald-300/80 text-[11px] leading-relaxed">
                99% of travelers don't need to change anything here. Fast PostgreSQL semantic caching and OpenRouter auto-routing are fully active.
              </span>
            </div>
          </div>
        )}

        {/* Active Form Fields for BYOK */}
        <div className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5 flex items-center justify-between">
              <span className="flex items-center space-x-1.5">
                <Key className="h-3.5 w-3.5 text-slate-400" />
                <span>Custom API Key (BYOK)</span>
              </span>
              <span className="text-[10px] text-slate-500">Stored in browser localStorage</span>
            </label>
            <input
              type="password"
              value={customApiKey}
              onChange={(e) => setCustomApiKey(e.target.value)}
              placeholder="sk-or-v1-... or sk-proj-..."
              className="w-full bg-slate-950 border border-slate-700 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 rounded-xl px-3 py-2 text-xs text-slate-200 placeholder-slate-600 outline-none transition font-mono"
            />
            <span className="text-[10px] text-slate-500 mt-1 block">
              Direct OpenRouter, OpenAI, Groq, or DeepSeek API key. Never saved on server disk.
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
              value={customModel}
              onChange={(e) => setCustomModel(e.target.value)}
              placeholder="openrouter/free (or e.g. anthropic/claude-3.5-sonnet, openai/gpt-4o)"
              className="w-full bg-slate-950 border border-slate-700 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 rounded-xl px-3 py-2 text-xs text-slate-200 placeholder-slate-600 outline-none transition font-mono"
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
              value={customBaseUrl}
              onChange={(e) => setCustomBaseUrl(e.target.value)}
              placeholder="https://openrouter.ai/api/v1 (or https://api.openai.com/v1)"
              className="w-full bg-slate-950 border border-slate-700 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 rounded-xl px-3 py-2 text-xs text-slate-200 placeholder-slate-600 outline-none transition font-mono"
            />
          </div>
        </div>

        {/* Modal Footer */}
        <div className="pt-3 border-t border-slate-800 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {(customApiKey || customModel || customBaseUrl) && (
              <button
                type="button"
                onClick={handleReset}
                className="inline-flex items-center space-x-1 px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-rose-500/20 text-slate-400 hover:text-rose-300 border border-slate-700/60 hover:border-rose-500/30 text-xs transition"
                title="Reset to default free server settings"
              >
                <RotateCcw className="h-3 w-3" />
                <span>Reset</span>
              </button>
            )}
            <span className="text-[10px] text-slate-500 hidden sm:flex items-center">
              <Sparkles className="h-3 w-3 mr-1 text-indigo-400" />
              Client Header Injection
            </span>
          </div>

          <div className="flex items-center space-x-2">
            <button
              type="button"
              onClick={onClose}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleSave}
              className="inline-flex items-center space-x-1.5 px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition shadow-sm"
            >
              {saved ? (
                <>
                  <Check className="h-3.5 w-3.5 text-emerald-300" />
                  <span>Saved!</span>
                </>
              ) : (
                <span>Save Settings</span>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
