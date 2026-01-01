import React, { useState, useEffect } from 'react';
import { GoogleGenAI } from '@google/genai';
import { Zap, FileText, CheckCheck, Download } from 'lucide-react';
import { saveToHistory, loadFromHistory } from '../utils/history';

interface FastLabState {
    input: string;
    output: string;
    mode: 'summarize' | 'grammar';
}

const HISTORY_KEY = 'textgpt_fastlab_state';

export const FastLab: React.FC = () => {
  // Load state from history
  const savedState = loadFromHistory<FastLabState>(HISTORY_KEY, { input: '', output: '', mode: 'summarize' });

  const [input, setInput] = useState(savedState.input);
  const [output, setOutput] = useState(savedState.output);
  const [mode, setMode] = useState<'summarize' | 'grammar'>(savedState.mode);
  const [loading, setLoading] = useState(false);

  // Save on any change
  useEffect(() => {
    saveToHistory(HISTORY_KEY, { input, output, mode });
  }, [input, output, mode]);

  const processText = async () => {
    if (!input) return;

    setLoading(true);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const prompt = mode === 'summarize' 
        ? `Summarize this text in 2 sentences:\n${input}`
        : `Fix grammar and make this text more professional:\n${input}`;

      // Use Flash Lite for speed
      const response = await ai.models.generateContent({
        model: 'gemini-flash-lite-latest',
        contents: prompt
      });
      
      setOutput(response.text || "No output");
    } catch (e) {
      setOutput("Error processing text.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!output) return;
    const blob = new Blob([output], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fast-lab-output-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="h-full bg-slate-900 p-6">
      <div className="max-w-2xl mx-auto space-y-6">
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <Zap className="text-yellow-400" />
          Fast Lab
        </h2>
        <p className="text-slate-400">High-speed tasks using TextGpt.</p>

        <div className="flex bg-slate-800 p-1 rounded-lg w-fit">
          <button
            onClick={() => setMode('summarize')}
            className={`px-4 py-2 rounded-md flex items-center gap-2 ${mode === 'summarize' ? 'bg-slate-600 text-white' : 'text-slate-400'}`}
          >
            <FileText size={16} /> Summarizer
          </button>
          <button
             onClick={() => setMode('grammar')}
             className={`px-4 py-2 rounded-md flex items-center gap-2 ${mode === 'grammar' ? 'bg-slate-600 text-white' : 'text-slate-400'}`}
          >
            <CheckCheck size={16} /> Proofreader
          </button>
        </div>

        <textarea 
          className="w-full h-40 bg-slate-800 border border-slate-700 rounded-lg p-4 text-white focus:ring-2 focus:ring-yellow-400 focus:outline-none"
          placeholder="Paste text here..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />

        <button
          onClick={processText}
          disabled={loading || !input}
          className="bg-yellow-500 hover:bg-yellow-600 disabled:opacity-50 text-slate-900 font-bold px-6 py-2 rounded-lg"
        >
          {loading ? 'Processing...' : 'Run Fast Action'}
        </button>

        {output && (
          <div className="bg-slate-800 border-l-4 border-yellow-400 p-4 rounded text-slate-200 relative group">
            <p className="whitespace-pre-wrap">{output}</p>
            <button 
                onClick={handleDownload}
                className="absolute top-2 right-2 p-2 bg-slate-700/50 hover:bg-slate-700 rounded text-slate-300 hover:text-white opacity-0 group-hover:opacity-100 transition-all"
                title="Download Result"
            >
                <Download size={16} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};