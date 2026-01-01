import React, { useState, useEffect } from 'react';
import { GoogleGenAI } from '@google/genai';
import { Eye, Upload, Loader2, ArrowRight, Download } from 'lucide-react';
import { saveToHistory, loadFromHistory } from '../utils/history';

const HISTORY_KEY = 'textgpt_vision_analysis';

export const VisionInterface: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  
  // Load last analysis from history
  const [analysis, setAnalysis] = useState<string>(() => loadFromHistory(HISTORY_KEY, ''));
  const [isLoading, setIsLoading] = useState(false);

  // Save analysis to history
  useEffect(() => {
    saveToHistory(HISTORY_KEY, analysis);
  }, [analysis]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
      setAnalysis('');
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile || !preview) return;

    setIsLoading(true);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const base64Data = preview.split(',')[1];
      
      const response = await ai.models.generateContent({
        model: 'gemini-3-pro-preview',
        contents: {
          parts: [
            {
              inlineData: {
                mimeType: selectedFile.type,
                data: base64Data
              }
            },
            {
              text: "Analyze this image in detail. Describe the objects, setting, and any text present."
            }
          ]
        }
      });
      
      setAnalysis(response.text || "No analysis returned.");
    } catch (error) {
      console.error(error);
      setAnalysis("Error analyzing image.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = () => {
    if (!analysis) return;
    const blob = new Blob([analysis], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vision-analysis-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="h-full bg-slate-900 p-6 overflow-y-auto">
      <div className="max-w-3xl mx-auto">
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
          <Eye className="text-emerald-400" />
          Vision Intelligence
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="border-2 border-dashed border-slate-600 rounded-xl p-8 text-center bg-slate-800/50 hover:bg-slate-800 transition-colors relative">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              {preview ? (
                <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-lg shadow-lg" />
              ) : (
                <div className="flex flex-col items-center text-slate-400">
                  <Upload size={48} className="mb-2" />
                  <p>Click or Drag to Upload Image</p>
                </div>
              )}
            </div>
            
            <button
              onClick={analyzeImage}
              disabled={!selectedFile || isLoading}
              className="w-full bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white py-3 rounded-lg font-bold flex justify-center items-center gap-2 transition-colors"
            >
              {isLoading ? <Loader2 className="animate-spin" /> : <ArrowRight />}
              Analyze with TextGpt
            </button>
          </div>

          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 min-h-[300px] flex flex-col">
             <div className="flex justify-between items-center mb-4">
               <h3 className="text-sm uppercase tracking-wider text-slate-500 font-bold">Analysis Result</h3>
               {analysis && (
                 <button 
                   onClick={handleDownload} 
                   className="text-slate-400 hover:text-emerald-400 transition-colors p-1"
                   title="Download Analysis"
                 >
                    <Download size={16} />
                 </button>
               )}
             </div>
             {isLoading ? (
               <div className="space-y-3 animate-pulse">
                 <div className="h-4 bg-slate-700 rounded w-3/4"></div>
                 <div className="h-4 bg-slate-700 rounded w-full"></div>
                 <div className="h-4 bg-slate-700 rounded w-5/6"></div>
               </div>
             ) : analysis ? (
               <div className="prose prose-invert prose-sm flex-1">
                 <p className="whitespace-pre-wrap text-slate-200 leading-relaxed">{analysis}</p>
               </div>
             ) : (
               <p className="text-slate-600 italic">Upload an image to see the magic...</p>
             )}
          </div>
        </div>
      </div>
    </div>
  );
};