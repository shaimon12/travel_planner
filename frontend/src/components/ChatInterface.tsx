import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  Send,
  Sparkles,
  MapPin,
  ExternalLink,
  Bot,
  User,
  Loader2,
  CloudSun,
  Lightbulb
} from 'lucide-react';
import { Message } from '../types';

interface ChatInterfaceProps {
  messages: Message[];
  onSendMessage: (content: string) => Promise<void>;
  isSending: boolean;
  destination?: string;
}

const QUICK_SUGGESTIONS = [
  'Recommend top hidden gem cafes ☕',
  'What is the current weather? ⛅',
  'Make Day 2 more relaxing 🧘',
  'Suggest best budget dinner spots 🍜',
];

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  onSendMessage,
  isSending,
  destination,
}) => {
  const [input, setInput] = useState('');
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of chat when new message arrives or sending changes
  const scrollToBottom = (behavior: ScrollBehavior = 'smooth') => {
    chatBottomRef.current?.scrollIntoView({ behavior, block: 'end' });
  };

  useEffect(() => {
    scrollToBottom('smooth');
  }, [messages, isSending]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isSending) return;
    const msg = input.trim();
    setInput('');
    await onSendMessage(msg);
  };

  const handleSuggestionClick = (suggestion: string) => {
    if (isSending) return;
    onSendMessage(suggestion);
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 overflow-hidden relative">
      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-400">
            <div className="h-16 w-16 rounded-2xl bg-indigo-600/10 border border-indigo-500/20 flex items-center justify-center mb-4">
              <Sparkles className="h-8 w-8 text-indigo-400" />
            </div>
            <h3 className="text-lg font-bold text-white mb-1">Your AI Travel Concierge</h3>
            <p className="text-xs text-slate-400 max-w-md mb-6">
              Powered by Google Gemini 2.5 Flash, Open-Meteo, and LangGraph multi-agent orchestration. Fill in your trip details to generate an itinerary!
            </p>
          </div>
        ) : (
          messages.map(msg => {
            const isUser = msg.role === 'user';
            return (
              <div
                key={msg.id}
                className={`flex items-start space-x-3 ${
                  isUser ? 'flex-row-reverse space-x-reverse' : 'flex-row'
                }`}
              >
                {/* Avatar */}
                <div
                  className={`h-8 w-8 rounded-xl flex items-center justify-center shrink-0 shadow-md ${
                    isUser
                      ? 'bg-indigo-600 text-white'
                      : 'bg-slate-800 border border-slate-700 text-sky-400'
                  }`}
                >
                  {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                </div>

                {/* Message Bubble */}
                <div
                  className={`max-w-2xl md:max-w-3xl rounded-2xl px-5 py-4 text-xs md:text-sm shadow-md ${
                    isUser
                      ? 'bg-indigo-600 text-white rounded-tr-sm'
                      : 'bg-slate-900 border border-slate-800 text-slate-200 rounded-tl-sm'
                  }`}
                >
                  {/* Weather banner if present in metadata */}
                  {!isUser && msg.meta?.weather_report && (
                    <div className="mb-3 p-2.5 rounded-lg bg-sky-500/10 border border-sky-500/20 flex items-start space-x-2 text-sky-200 text-xs">
                      <CloudSun className="h-4 w-4 text-sky-400 shrink-0 mt-0.5" />
                      <div className="whitespace-pre-line leading-relaxed font-mono text-[11px]">
                        {msg.meta.weather_report}
                      </div>
                    </div>
                  )}

                  {/* Render content */}
                  {isUser ? (
                    <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                  ) : (
                    <div className="markdown-body leading-relaxed space-y-2">
                      <ReactMarkdown
                        components={{
                          a: ({ href, children }) => {
                            const isGoogleMaps = href?.includes('google.com/maps');
                            return (
                              <a
                                href={href}
                                target="_blank"
                                rel="noopener noreferrer"
                                className={`inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium transition ${
                                  isGoogleMaps
                                    ? 'bg-indigo-500/15 text-indigo-300 hover:bg-indigo-500/25 border border-indigo-500/30'
                                    : 'text-sky-400 hover:underline'
                                }`}
                              >
                                {isGoogleMaps && <MapPin className="h-3 w-3 mr-1 text-indigo-400 shrink-0" />}
                                <span>{children}</span>
                                <ExternalLink className="h-2.5 w-2.5 ml-1 opacity-70 shrink-0" />
                              </a>
                            );
                          },
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  )}

                  {/* Timestamp */}
                  <div
                    className={`mt-2 text-[10px] ${
                      isUser ? 'text-indigo-200/70 text-right' : 'text-slate-500'
                    }`}
                  >
                    {new Date(msg.created_at).toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </div>
                </div>
              </div>
            );
          })
        )}

        {/* Loading Bubble */}
        {isSending && (
          <div className="flex items-start space-x-3">
            <div className="h-8 w-8 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center text-sky-400">
              <Bot className="h-4 w-4" />
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-sm px-4 py-3 flex items-center space-x-2 text-xs text-slate-300">
              <Loader2 className="h-4 w-4 animate-spin text-indigo-400" />
              <span>Travel Concierge is analyzing your request...</span>
            </div>
          </div>
        )}

        {/* Native Auto-Scroll Anchor */}
        <div ref={chatBottomRef} id="chat-bottom" className="h-2" />
      </div>

      {/* Input Section Pinned to Bottom */}
      <div className="border-t border-slate-800 bg-slate-900/80 backdrop-blur p-4 space-y-3">
        {/* Quick Suggestion Pills */}
        <div className="flex items-center space-x-2 overflow-x-auto pb-1 text-xs">
          <span className="text-[11px] text-slate-500 flex items-center shrink-0">
            <Lightbulb className="h-3 w-3 mr-1 text-amber-400" /> Tips:
          </span>
          {QUICK_SUGGESTIONS.map(s => (
            <button
              key={s}
              onClick={() => handleSuggestionClick(s)}
              disabled={isSending}
              className="px-2.5 py-1 rounded-full bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700/80 text-[11px] shrink-0 transition disabled:opacity-50"
            >
              {s}
            </button>
          ))}
        </div>

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="flex items-center space-x-2">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            disabled={isSending}
            placeholder={
              destination
                ? `Ask for changes or recommendations in ${destination}... (e.g. 'Where can I get boba tea?')`
                : 'Plan a trip to unlock concierge chat...'
            }
            className="flex-1 bg-slate-950 border border-slate-700/80 rounded-xl px-4 py-2.5 text-xs md:text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 shadow-inner"
          />

          <button
            type="submit"
            disabled={!input.trim() || isSending}
            className="px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs md:text-sm flex items-center space-x-1.5 shadow-md shadow-indigo-600/30 transition disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isSending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <>
                <span>Send</span>
                <Send className="h-3.5 w-3.5" />
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
};
