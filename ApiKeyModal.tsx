import React, { useState, useEffect } from 'react';
import { Key, X, Check, ExternalLink } from 'lucide-react';
import { getApiKey, setApiKey, hasValidKey } from '../utils/apiKey';

interface ApiKeyModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export const ApiKeyModal: React.FC<ApiKeyModalProps> = ({ isOpen, onClose, onSuccess }) => {
  const [key, setKey] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    if (isOpen) {
      setKey(getApiKey());
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSave = () => {
    if (!key.trim()) {
      setError('API Key cannot be empty');
      return;
    }
    setApiKey(key.trim());
    onSuccess();
    onClose();
  };

  return (
    <div className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 w-full max-w-md rounded-2xl shadow-2xl overflow-hidden">
        <div className="p-6 space-y-6">
          <div className="flex justify-between items-start">
            <div className="space-y-1">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <Key className="text-indigo-400" size={24} />
                API Key Required
              </h2>
              <p className="text-sm text-slate-400">
                Enter your Google Gemini API key to continue.
              </p>
            </div>
            {hasValidKey() && (
              <button onClick={onClose} className="text-slate-500 hover:text-white">
                <X size={20} />
              </button>
            )}
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-500 uppercase">Your API Key</label>
              <input
                type="password"
                value={key}
                onChange={(e) => {
                  setKey(e.target.value);
                  setError('');
                }}
                placeholder="AIzaSy..."
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition-all"
              />
              {error && <p className="text-red-400 text-xs">{error}</p>}
            </div>

            <button
              onClick={handleSave}
              className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 rounded-lg flex items-center justify-center gap-2 transition-colors"
            >
              <Check size={18} />
              Save & Continue
            </button>

            <div className="pt-4 border-t border-slate-800 text-center">
              <a 
                href="https://aistudio.google.com/app/apikey" 
                target="_blank" 
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300"
              >
                Get a free API Key from Google AI Studio <ExternalLink size={10} />
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};