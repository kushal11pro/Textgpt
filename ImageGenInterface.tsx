import React, { useState, useEffect } from 'react';
import { GoogleGenAI } from '@google/genai';
import { Image, Download, Zap, Key, Clock, Trash2 } from 'lucide-react';
import { ImageResolution } from '../types';
import { saveToHistory, loadFromHistory, clearHistory } from '../utils/history';

interface HistoryItem {
    id: number;
    prompt: string;
    image: string;
    timestamp: number;
}

const HISTORY_KEY = 'textgpt_image_history';

// Removed conflicting global declaration of aistudio to avoid TypeScript errors.
// Accessing window.aistudio via type assertion (window as any).aistudio instead.

export const ImageGenInterface: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [resolution, setResolution] = useState<ImageResolution>(ImageResolution.RES_1K);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [history, setHistory] = useState<HistoryItem[]>(() => loadFromHistory(HISTORY_KEY, []));

  useEffect(() => {
    saveToHistory(HISTORY_KEY, history);
  }, [history]);

  const handleGenerate = async () => {
    if (!prompt) return;

    setIsLoading(true);
    setError(null);
    setGeneratedImage(null);

    try {
      let model = 'gemini-2.5-flash-image';
      let imageConfig: any = {
        aspectRatio: '1:1',
      };

      // High-quality image gen requires user-selected key per guidelines
      // Also switch to Pro model for high res
      if (resolution === ImageResolution.RES_2K || resolution === ImageResolution.RES_4K) {
        model = 'gemini-3-pro-image-preview';
        imageConfig.imageSize = resolution;

        const win = window as any;
        if (win.aistudio) {
          const hasKey = await win.aistudio.hasSelectedApiKey();
          if (!hasKey) {
            await win.aistudio.openSelectKey();
          }
        }
      }

      // We re-instantiate here to ensure we pick up any potentially selected key if the environment handles it that way
      // But primarily we rely on process.env.API_KEY for the 'code' structure, assuming the environment
      // injects the selected key into process.env.API_KEY if the user selects it via the window.aistudio flow.
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      const response = await ai.models.generateContent({
        model: model,
        contents: {
          parts: [{ text: prompt }],
        },
        config: {
          imageConfig: imageConfig,
        },
      });

      let foundImage = false;
      if (response.candidates?.[0]?.content?.parts) {
        for (const part of response.candidates[0].content.parts) {
          if (part.inlineData) {
            const base64Str = part.inlineData.data;
            const dataUrl = `data:image/png;base64,${base64Str}`;
            setGeneratedImage(dataUrl);
            foundImage = true;

            // Add to history (Limit to last 3 to save space/memory)
            setHistory(prev => {
                const newItem = { id: Date.now(), prompt, image: dataUrl, timestamp: Date.now() };
                const newHistory = [newItem, ...prev].slice(0, 3);
                return newHistory;
            });

            break;
          }
        }
      }

      if (!foundImage) {
        setError('No image generated. The model might have refused the prompt.');
      }

    } catch (err: any) {
      console.error(err);
      if (err.message && err.message.includes('Requested entity was not found')) {
         // Handle race condition/key reset as per prompt
         const win = window as any;
         if (win.aistudio) {
            await win.aistudio.openSelectKey();
            setError("Key selection update required. Please try again.");
         }
      } else {
        setError("Generation failed. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const clearImageHistory = () => {
      setHistory([]);
      clearHistory(HISTORY_KEY);
  };

  return (
    <div className="h-full bg-slate-900 flex flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto space-y-8">
            <div className="space-y-2">
            <h2 className="text-3xl font-bold text-white flex items-center gap-2">
                <Image className="text-pink-500" />
                Studio Pro
            </h2>
            <p className="text-slate-400">Generate high-fidelity images (up to 4K) using TextGpt.</p>
            </div>

            <div className="bg-slate-800 p-6 rounded-2xl border border-slate-700 space-y-6">
            <div className="space-y-2">
                <label className="text-sm font-medium text-slate-300">Prompt</label>
                <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="A futuristic city made of crystal, golden hour lighting, cinematic..."
                className="w-full h-32 bg-slate-900 border border-slate-700 rounded-lg p-4 text-white focus:ring-2 focus:ring-pink-500 focus:outline-none resize-none"
                />
            </div>

            <div className="flex flex-wrap gap-6 items-center">
                <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-300">Resolution</label>
                    <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-700">
                    {Object.values(ImageResolution).map((res) => (
                        <button
                        key={res}
                        onClick={() => setResolution(res)}
                        className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                            resolution === res
                            ? 'bg-pink-600 text-white shadow-lg'
                            : 'text-slate-400 hover:text-white hover:bg-slate-800'
                        }`}
                        >
                        {res}
                        </button>
                    ))}
                    </div>
                </div>
                
                <div className="flex-1 flex justify-end items-end h-full">
                    <button
                    onClick={handleGenerate}
                    disabled={isLoading || !prompt}
                    className="bg-pink-600 hover:bg-pink-700 disabled:opacity-50 text-white px-8 py-3 rounded-lg font-bold flex items-center gap-2 transition-all shadow-lg hover:shadow-pink-500/20"
                    >
                    {isLoading ? (
                        <>Generating...</>
                    ) : (
                        <>
                        <Zap size={20} />
                        Generate
                        </>
                    )}
                    </button>
                </div>
            </div>
            
            {(resolution === ImageResolution.RES_2K || resolution === ImageResolution.RES_4K) && (
                <p className="text-xs text-amber-400/80 flex items-center gap-1">
                <Key size={12} /> High-res generation requires a paid API key selection.
                </p>
            )}
            </div>

            {error && (
            <div className="bg-red-900/20 border border-red-500/50 text-red-200 p-4 rounded-lg">
                {error}
            </div>
            )}

            {generatedImage && (
            <div className="bg-slate-800 rounded-2xl p-2 border border-slate-700 shadow-2xl">
                <img
                src={generatedImage}
                alt="Generated"
                className="w-full h-auto rounded-xl"
                />
                <div className="p-4 flex justify-end">
                <a
                    href={generatedImage}
                    download={`textgpt-gen-${Date.now()}.png`}
                    className="flex items-center gap-2 text-slate-300 hover:text-white transition-colors"
                >
                    <Download size={20} />
                    Download PNG
                </a>
                </div>
            </div>
            )}
        </div>
      </div>

      {/* History Panel */}
      {history.length > 0 && (
          <div className="border-t border-slate-800 bg-slate-900 p-4 shrink-0">
             <div className="max-w-4xl mx-auto">
                 <div className="flex justify-between items-center mb-2">
                     <h3 className="text-xs font-bold uppercase text-slate-500 flex items-center gap-1">
                         <Clock size={12} /> Recent History
                     </h3>
                     <button onClick={clearImageHistory} className="text-slate-600 hover:text-red-400">
                         <Trash2 size={12} />
                     </button>
                 </div>
                 <div className="flex gap-4 overflow-x-auto pb-2">
                     {history.map(item => (
                         <div key={item.id} className="relative group shrink-0 w-32 cursor-pointer" onClick={() => { setGeneratedImage(item.image); setPrompt(item.prompt); }}>
                             <img src={item.image} alt="History" className="w-32 h-32 object-cover rounded-lg border border-slate-700" />
                             <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center">
                                 <span className="text-xs text-white p-1 text-center line-clamp-2">{item.prompt}</span>
                             </div>
                         </div>
                     ))}
                 </div>
             </div>
          </div>
      )}
    </div>
  );
};