import React, { useState, useEffect } from 'react';
import { GoogleGenAI } from '@google/genai';
import { Video, Clapperboard, Download, Loader2, Clock, Trash2, AlertTriangle } from 'lucide-react';
import { saveToHistory, loadFromHistory, clearHistory } from '../utils/history';

interface VideoHistoryItem {
    id: number;
    prompt: string;
    videoUri: string;
    timestamp: number;
}

const HISTORY_KEY = 'textgpt_video_history';

export const VideoGenInterface: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');
  const [generatedVideo, setGeneratedVideo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useHighQuality, setUseHighQuality] = useState(false);
  
  const [history, setHistory] = useState<VideoHistoryItem[]>(() => loadFromHistory(HISTORY_KEY, []));

  useEffect(() => {
    saveToHistory(HISTORY_KEY, history);
  }, [history]);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;

    setIsGenerating(true);
    setError(null);
    setGeneratedVideo(null);
    setLoadingStatus('Initializing Veo model...');

    try {
      // Veo requires a paid key, ensure we handle the selection flow if available
      const win = window as any;
      if (win.aistudio) {
        const hasKey = await win.aistudio.hasSelectedApiKey();
        if (!hasKey) {
          await win.aistudio.openSelectKey();
        }
      }

      // Create new instance with latest key
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      const model = useHighQuality ? 'veo-3.1-generate-preview' : 'veo-3.1-fast-generate-preview';
      
      setLoadingStatus('Submitting generation request...');
      
      let operation = await ai.models.generateVideos({
        model: model,
        prompt: prompt,
        config: {
          numberOfVideos: 1,
          resolution: '720p', // Veo supports 720p or 1080p
          aspectRatio: '16:9'
        }
      });

      setLoadingStatus('Rendering video (this may take a minute)...');

      // Polling loop
      while (!operation.done) {
        await new Promise(resolve => setTimeout(resolve, 5000)); // Check every 5 seconds
        setLoadingStatus('Still rendering...');
        operation = await ai.operations.getVideosOperation({operation: operation});
      }

      if (operation.error) {
         throw new Error((operation.error as any).message || "Video generation failed during processing.");
      }

      const videoUri = operation.response?.generatedVideos?.[0]?.video?.uri;
      
      if (videoUri) {
          // IMPORTANT: The URI requires the API key to be appended to fetch/play
          // We don't download the bytes here to keep it light, we just point the video tag to it.
          // Note: This exposes the key in the DOM src attribute, which is standard for this client-side demo context.
          const authenticatedUri = `${videoUri}&key=${process.env.API_KEY}`;
          
          setGeneratedVideo(authenticatedUri);
          
          setHistory(prev => {
             const newItem = { id: Date.now(), prompt, videoUri: authenticatedUri, timestamp: Date.now() };
             return [newItem, ...prev].slice(0, 5); // Keep last 5
          });
      } else {
          throw new Error("No video URI returned from the API.");
      }

    } catch (err: any) {
      console.error(err);
      if (err.message && err.message.includes('Requested entity was not found')) {
         const win = window as any;
         if (win.aistudio) {
            await win.aistudio.openSelectKey();
            setError("Key selection update required. Please try again.");
         }
      } else {
        setError(err.message || "Generation failed. Please try again.");
      }
    } finally {
      setIsGenerating(false);
      setLoadingStatus('');
    }
  };

  const clearVideoHistory = () => {
    if (window.confirm("Clear video history?")) {
        setHistory([]);
        clearHistory(HISTORY_KEY);
    }
  };

  return (
    <div className="h-full bg-slate-900 flex flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="space-y-2">
            <h2 className="text-3xl font-bold text-white flex items-center gap-2">
              <Video className="text-purple-500" />
              Veo Studio
            </h2>
            <p className="text-slate-400">Generate 1080p videos from text using Google's Veo model.</p>
          </div>

          <div className="bg-slate-800 p-6 rounded-2xl border border-slate-700 space-y-6">
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-300">Prompt</label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="A cinematic drone shot of a futuristic cyberpunk city in the rain, neon lights reflecting on wet pavement..."
                className="w-full h-32 bg-slate-900 border border-slate-700 rounded-lg p-4 text-white focus:ring-2 focus:ring-purple-500 focus:outline-none resize-none"
              />
            </div>

            <div className="flex items-center justify-between">
               <div className="flex items-center gap-2">
                   <input 
                     type="checkbox" 
                     id="hq" 
                     checked={useHighQuality} 
                     onChange={(e) => setUseHighQuality(e.target.checked)}
                     className="w-4 h-4 rounded border-slate-600 text-purple-600 focus:ring-purple-500 bg-slate-700"
                   />
                   <label htmlFor="hq" className="text-sm text-slate-300 cursor-pointer select-none">
                       High Quality Mode (Slower)
                   </label>
               </div>

               <button
                  onClick={handleGenerate}
                  disabled={isGenerating || !prompt}
                  className="bg-purple-600 hover:bg-purple-700 disabled:opacity-50 text-white px-8 py-3 rounded-lg font-bold flex items-center gap-2 transition-all shadow-lg hover:shadow-purple-500/20"
                >
                  {isGenerating ? (
                    <>
                       <Loader2 className="animate-spin" size={20} />
                       Generating...
                    </>
                  ) : (
                    <>
                      <Clapperboard size={20} />
                      Generate Video
                    </>
                  )}
                </button>
            </div>
            
             <p className="text-xs text-amber-400/80 flex items-center gap-1">
               <AlertTriangle size={12} /> Veo requires a paid API key project. Generation may take 1-2 minutes.
             </p>
          </div>

          {isGenerating && (
             <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-8 flex flex-col items-center justify-center text-center space-y-4 animate-pulse">
                 <Loader2 className="animate-spin text-purple-500" size={48} />
                 <p className="text-slate-300 font-medium">{loadingStatus}</p>
                 <p className="text-xs text-slate-500">Please do not close this tab.</p>
             </div>
          )}

          {error && (
            <div className="bg-red-900/20 border border-red-500/50 text-red-200 p-4 rounded-lg">
              {error}
            </div>
          )}

          {generatedVideo && (
            <div className="bg-slate-800 rounded-2xl p-2 border border-slate-700 shadow-2xl space-y-2">
               <div className="relative aspect-video bg-black rounded-xl overflow-hidden">
                  <video 
                    src={generatedVideo} 
                    controls 
                    autoPlay 
                    loop 
                    className="w-full h-full object-contain"
                  />
               </div>
               <div className="p-2 flex justify-end">
                  <a 
                    href={generatedVideo} 
                    download={`veo-generation-${Date.now()}.mp4`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-slate-300 hover:text-white text-sm"
                  >
                      <Download size={16} /> Download MP4
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
                         <Clock size={12} /> Video History
                     </h3>
                     <button onClick={clearVideoHistory} className="text-slate-600 hover:text-red-400">
                         <Trash2 size={12} />
                     </button>
                 </div>
                 <div className="flex gap-4 overflow-x-auto pb-2">
                     {history.map(item => (
                         <div key={item.id} className="relative group shrink-0 w-48 cursor-pointer" onClick={() => { setGeneratedVideo(item.videoUri); setPrompt(item.prompt); }}>
                             <div className="w-48 h-28 bg-black rounded-lg border border-slate-700 flex items-center justify-center overflow-hidden relative">
                                <video src={item.videoUri} className="w-full h-full object-cover opacity-50 group-hover:opacity-100 transition-opacity" muted />
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <Clapperboard className="text-white opacity-80" />
                                </div>
                             </div>
                             <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center pointer-events-none">
                                 <span className="text-xs text-white p-2 text-center line-clamp-3">{item.prompt}</span>
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