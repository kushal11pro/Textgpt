import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenAI, Schema, Type } from '@google/genai';
import { Play, Code, Loader2, FileCode, Copy, RefreshCw, Send, Terminal, Maximize2, Minimize2, FolderPlus, Trash2, Plus, X, File, Download } from 'lucide-react';
import { CodeFile } from '../types';
import { getApiKey } from '../utils/apiKey';
import { saveToHistory, loadFromHistory, clearHistory } from '../utils/history';

interface CodeWorkspaceProps {
  initialPrompt?: string;
  onClearInitialPrompt?: () => void;
}

const FILES_KEY = 'textgpt_code_files';
const CHAT_KEY = 'textgpt_code_chat';

export const CodeWorkspace: React.FC<CodeWorkspaceProps> = ({ initialPrompt, onClearInitialPrompt }) => {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  
  // Load initial state from history
  const [files, setFiles] = useState<CodeFile[]>(() => loadFromHistory(FILES_KEY, []));
  const [activeFile, setActiveFile] = useState<string | null>(null);
  const [previewKey, setPreviewKey] = useState(0);
  const [chatHistory, setChatHistory] = useState<{role: 'user' | 'model', text: string}[]>(() => loadFromHistory(CHAT_KEY, []));
  const [isFullPreview, setIsFullPreview] = useState(false);
  
  // File Structure Modal State
  const [showFileModal, setShowFileModal] = useState(false);
  const [newFilename, setNewFilename] = useState('');

  const scrollRef = useRef<HTMLDivElement>(null);
  const hasAutoRunRef = useRef(false);

  // Save changes to history
  useEffect(() => {
    saveToHistory(FILES_KEY, files);
  }, [files]);

  useEffect(() => {
    saveToHistory(CHAT_KEY, chatHistory);
  }, [chatHistory]);

  // Set active file on load if exists
  useEffect(() => {
    if (files.length > 0 && !activeFile) {
        setActiveFile(files[0].filename);
    }
  }, []);

  // Scroll to bottom of chat
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [chatHistory]);

  // Handle incoming redirect prompt
  useEffect(() => {
    if (initialPrompt && !hasAutoRunRef.current && !isGenerating) {
      hasAutoRunRef.current = true;
      setPrompt(initialPrompt);
      // Auto-trigger generation
      handleGenerate(initialPrompt);
      if (onClearInitialPrompt) onClearInitialPrompt();
    }
  }, [initialPrompt]);

  const handleResetProject = () => {
    if (window.confirm("Are you sure you want to reset the entire project? This cannot be undone.")) {
        setFiles([]);
        setChatHistory([]);
        setActiveFile(null);
        clearHistory(FILES_KEY);
        clearHistory(CHAT_KEY);
    }
  };

  const handleGenerate = async (overridePrompt?: string) => {
    const textToUse = overridePrompt || prompt;
    if (!textToUse.trim()) return;
    
    const apiKey = getApiKey();
    if (!apiKey) {
      alert("API Key required");
      return;
    }

    setIsGenerating(true);
    setPrompt('');
    
    setChatHistory(prev => [...prev, { role: 'user', text: textToUse }]);

    try {
      const ai = new GoogleGenAI({ apiKey });
      
      const responseSchema: Schema = {
        type: Type.OBJECT,
        properties: {
          explanation: { type: Type.STRING },
          files: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                filename: { type: Type.STRING },
                content: { type: Type.STRING },
                language: { type: Type.STRING }
              },
              required: ["filename", "content", "language"]
            }
          }
        },
        required: ["files", "explanation"]
      };

      // Construct prompt with awareness of existing file structure
      let systemInstruction = "You are an expert full-stack developer. Always output code in a structured JSON format containing a list of files. Ensure code is complete, functional, and production-ready.";
      
      let contextPrompt = `Create code for: ${textToUse}. Generate multiple files if needed (e.g. index.html, styles.css, script.js). Return JSON.`;
      
      if (files.length > 0) {
         contextPrompt += `\n\nCurrent File Structure: ${files.map(f => f.filename).join(', ')}. \nUpdate these files or create new ones as needed to fulfill the request. Preserve existing functionality unless asked to change it.`;
      }

      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: [
            ...chatHistory.map(m => ({ role: m.role, parts: [{ text: m.text }] })),
            { role: 'user', parts: [{ text: contextPrompt }] }
        ],
        config: {
          responseMimeType: 'application/json',
          responseSchema: responseSchema,
          systemInstruction: systemInstruction
        }
      });

      const jsonStr = response.text;
      if (jsonStr) {
        const result = JSON.parse(jsonStr);
        setChatHistory(prev => [...prev, { role: 'model', text: result.explanation || "Code generated successfully." }]);
        
        if (result.files && Array.isArray(result.files)) {
            // Smart Merge: Update existing files or add new ones
            setFiles(prev => {
                const newFilesMap = new Map(prev.map(f => [f.filename, f]));
                result.files.forEach((f: CodeFile) => {
                    newFilesMap.set(f.filename, f);
                });
                return Array.from(newFilesMap.values());
            });

            if (result.files.length > 0) {
                // Switch to the first modified/created file
                setActiveFile(result.files[0].filename);
            }
            setPreviewKey(prev => prev + 1);
        }
      }

    } catch (error) {
      console.error(error);
      setChatHistory(prev => [...prev, { role: 'model', text: "Error generating code. Please try again." }]);
    } finally {
      setIsGenerating(false);
    }
  };

  const generatePreviewHtml = () => {
    const indexHtml = files.find(f => f.filename === 'index.html' || f.filename.endsWith('.html'));
    if (!indexHtml) return '<div style="color:#94a3b8; padding:20px; font-family:sans-serif;">No index.html found.<br/>Create one using the Files button or ask the AI to generate it.</div>';

    let htmlContent = indexHtml.content;

    // Inject CSS
    const cssFiles = files.filter(f => f.language === 'css' || f.filename.endsWith('.css'));
    let cssBlock = '';
    cssFiles.forEach(f => {
      cssBlock += `<style>\n${f.content}\n</style>\n`;
    });
    
    // Inject JS
    const jsFiles = files.filter(f => f.language === 'javascript' || f.filename.endsWith('.js'));
    let jsBlock = '';
    jsFiles.forEach(f => {
       jsBlock += `<script>\n${f.content}\n</script>\n`;
    });

    // Simple injection into head/body
    if (htmlContent.includes('</head>')) {
        htmlContent = htmlContent.replace('</head>', `${cssBlock}</head>`);
    } else {
        htmlContent = `${cssBlock}${htmlContent}`;
    }

    if (htmlContent.includes('</body>')) {
        htmlContent = htmlContent.replace('</body>', `${jsBlock}</body>`);
    } else {
        htmlContent = `${htmlContent}${jsBlock}`;
    }

    return htmlContent;
  };

  const handleCodeChange = (newContent: string) => {
    if (!activeFile) return;
    setFiles(prev => prev.map(f => 
      f.filename === activeFile ? { ...f, content: newContent } : f
    ));
  };

  const handleAddFile = () => {
      if (!newFilename.trim()) return;
      if (files.some(f => f.filename === newFilename.trim())) {
          alert('File already exists');
          return;
      }
      // Simple extension detection
      const ext = newFilename.split('.').pop()?.toLowerCase();
      let lang = 'text';
      if (ext === 'html') lang = 'html';
      else if (ext === 'css') lang = 'css';
      else if (ext === 'js') lang = 'javascript';
      else if (ext === 'ts' || ext === 'tsx') lang = 'typescript';
      else if (ext === 'jsx') lang = 'javascript';
      else if (ext === 'json') lang = 'json';

      const newFile: CodeFile = {
          filename: newFilename.trim(),
          content: '',
          language: lang
      };
      setFiles(prev => [...prev, newFile]);
      setNewFilename('');
      setActiveFile(newFile.filename);
  };

  const handleDeleteFile = (fname: string) => {
      setFiles(prev => prev.filter(f => f.filename !== fname));
      if (activeFile === fname) {
          setActiveFile(null);
      }
  };

  const handleDownloadActiveFile = () => {
    if (!activeFileContent) return;
    const blob = new Blob([activeFileContent.content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = activeFileContent.filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const activeFileContent = files.find(f => f.filename === activeFile);

  return (
    <div className="flex h-full bg-slate-950 text-slate-200 overflow-hidden relative">
      
      {/* File Structure Modal */}
      {showFileModal && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
            <div className="bg-slate-900 border border-slate-700 w-full max-w-md rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[80vh]">
                <div className="p-4 border-b border-slate-700 flex justify-between items-center bg-slate-800">
                    <h3 className="font-bold text-white flex items-center gap-2">
                        <FolderPlus size={18} className="text-indigo-400"/>
                        File Structure
                    </h3>
                    <button onClick={() => setShowFileModal(false)} className="text-slate-400 hover:text-white">
                        <X size={20} />
                    </button>
                </div>
                
                <div className="flex-1 overflow-y-auto p-2 space-y-1">
                    {files.length === 0 && (
                        <div className="text-center py-8 text-slate-500 text-sm">
                            No files yet. Add one below.
                        </div>
                    )}
                    {files.map(f => (
                        <div key={f.filename} className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg group hover:bg-slate-800 border border-transparent hover:border-slate-700">
                            <div className="flex items-center gap-3">
                                <FileCode size={16} className="text-slate-400" />
                                <span className="text-sm font-medium text-slate-200">{f.filename}</span>
                            </div>
                            <button 
                                onClick={() => handleDeleteFile(f.filename)}
                                className="text-slate-600 hover:text-red-400 p-1 opacity-0 group-hover:opacity-100 transition-all"
                                title="Delete File"
                            >
                                <Trash2 size={16} />
                            </button>
                        </div>
                    ))}
                </div>

                <div className="p-4 border-t border-slate-700 bg-slate-800">
                    <div className="flex gap-2">
                        <input 
                            type="text" 
                            value={newFilename}
                            onChange={(e) => setNewFilename(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleAddFile()}
                            placeholder="e.g. styles.css"
                            className="flex-1 bg-slate-900 border border-slate-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500"
                        />
                        <button 
                            onClick={handleAddFile}
                            disabled={!newFilename.trim()}
                            className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white p-2 rounded transition-colors"
                        >
                            <Plus size={20} />
                        </button>
                    </div>
                </div>
            </div>
        </div>
      )}

      {/* Sidebar / Chat Panel */}
      <div className="w-80 flex-none border-r border-slate-800 flex flex-col bg-slate-900 hidden md:flex">
        <div className="p-4 border-b border-slate-800 font-bold flex items-center justify-between text-indigo-400">
            <div className="flex items-center gap-2">
                <Terminal size={20} />
                Code Studio
            </div>
            {files.length > 0 && (
                <button 
                    onClick={handleResetProject} 
                    className="text-slate-600 hover:text-red-400 transition-colors"
                    title="Reset Project"
                >
                    <Trash2 size={16} />
                </button>
            )}
        </div>
        
        <div className="flex-1 overflow-y-auto p-4 space-y-4" ref={scrollRef}>
             {chatHistory.length === 0 && (
                 <div className="text-center text-slate-500 mt-10">
                     <FileCode size={40} className="mx-auto mb-4 opacity-50" />
                     <p>Describe the app or feature you want to build.</p>
                 </div>
             )}
             {chatHistory.map((msg, idx) => (
                 <div key={idx} className={`p-3 rounded-lg text-sm ${msg.role === 'user' ? 'bg-indigo-600/20 text-indigo-100 ml-8' : 'bg-slate-800 text-slate-300 mr-8'}`}>
                     <span className="font-bold text-xs uppercase opacity-50 mb-1 block">{msg.role}</span>
                     {msg.text}
                 </div>
             ))}
             {isGenerating && (
                 <div className="flex items-center gap-2 text-slate-400 text-sm">
                     <Loader2 className="animate-spin" size={14} />
                     Generating...
                 </div>
             )}
        </div>

        <div className="p-4 border-t border-slate-800 bg-slate-900">
            <div className="relative">
                <input
                    type="text"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleGenerate()}
                    placeholder="Describe changes..."
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg pl-4 pr-10 py-3 text-sm focus:outline-none focus:border-indigo-500"
                />
                <button 
                    onClick={() => handleGenerate()}
                    disabled={isGenerating || !prompt.trim()}
                    className="absolute right-2 top-2 p-1.5 bg-indigo-600 hover:bg-indigo-500 rounded text-white disabled:opacity-50"
                >
                    <Send size={16} />
                </button>
            </div>
        </div>
      </div>

      {/* Main Editor & Preview Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
         {/* Top Bar: Tabs & Actions */}
         <div className="h-12 border-b border-slate-800 bg-slate-900 flex items-center justify-between px-2">
             <div className="flex items-center gap-1 overflow-x-auto flex-1 no-scrollbar">
                 {files.map(file => (
                     <button
                        key={file.filename}
                        onClick={() => setActiveFile(file.filename)}
                        className={`px-4 py-2 text-xs font-medium border-t-2 transition-colors flex items-center gap-2 whitespace-nowrap ${activeFile === file.filename ? 'border-indigo-500 bg-slate-800 text-white' : 'border-transparent text-slate-500 hover:text-slate-300'}`}
                     >
                        <Code size={12} />
                        {file.filename}
                     </button>
                 ))}
                 {files.length === 0 && (
                     <span className="text-slate-600 text-xs px-4">No active files</span>
                 )}
             </div>

             {/* Action Buttons */}
             <div className="flex items-center gap-2 ml-2 pl-2 border-l border-slate-800">
                 <button 
                     onClick={() => setShowFileModal(true)}
                     className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white hover:bg-slate-800 rounded transition-colors whitespace-nowrap"
                     title="Manage Files"
                 >
                     <FolderPlus size={14} />
                     <span>Files</span>
                 </button>
                 
                 <button 
                     onClick={handleDownloadActiveFile}
                     disabled={!activeFileContent}
                     className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white hover:bg-slate-800 rounded transition-colors whitespace-nowrap disabled:opacity-50"
                     title="Download Active File"
                 >
                     <Download size={14} />
                 </button>
             </div>
         </div>

         {/* Content Area */}
         <div className="flex-1 flex flex-col md:flex-row overflow-hidden relative">
             {/* Code Editor */}
             <div className={`bg-slate-950 overflow-y-auto relative border-r border-slate-800 ${isFullPreview ? 'hidden' : 'flex-1'}`}>
                 {activeFileContent ? (
                     <div className="p-0 h-full relative group">
                        <textarea 
                            className="w-full h-full bg-slate-950 text-emerald-300 font-mono text-sm p-4 resize-none focus:outline-none leading-relaxed"
                            value={activeFileContent.content}
                            onChange={(e) => handleCodeChange(e.target.value)}
                            spellCheck={false}
                        />
                        <button 
                            onClick={() => navigator.clipboard.writeText(activeFileContent.content)}
                            className="absolute top-4 right-4 p-2 bg-slate-800 rounded opacity-0 group-hover:opacity-100 transition-opacity hover:text-white text-slate-400 shadow-lg"
                            title="Copy code"
                        >
                            <Copy size={16} />
                        </button>
                     </div>
                 ) : (
                     <div className="flex flex-col items-center justify-center h-full text-slate-600 gap-4">
                         <FileCode size={48} className="opacity-20" />
                         <p>Select a file to edit or create a new one.</p>
                         <button 
                            onClick={() => setShowFileModal(true)}
                            className="text-indigo-400 hover:text-indigo-300 text-sm flex items-center gap-2"
                         >
                             <Plus size={14} /> Create File
                         </button>
                     </div>
                 )}
             </div>

             {/* Preview Pane */}
             {(files.length > 0 || isFullPreview) && (
                 <div className={`flex flex-col bg-slate-100 border-l border-slate-300 transition-all duration-300 ${isFullPreview ? 'w-full flex-1' : 'w-full md:w-1/2'}`}>
                     <div className="h-8 bg-slate-200 border-b border-slate-300 flex items-center justify-between px-3">
                         <span className="text-slate-600 text-xs font-bold uppercase flex items-center gap-1">
                             <Play size={10} /> Live Preview
                         </span>
                         <div className="flex items-center gap-2">
                             <button 
                                onClick={() => setIsFullPreview(!isFullPreview)} 
                                className="text-slate-500 hover:text-slate-800"
                                title={isFullPreview ? "Exit Full Screen" : "Full Screen Preview"}
                             >
                                 {isFullPreview ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
                             </button>
                             <button onClick={() => setPreviewKey(k => k+1)} className="text-slate-500 hover:text-slate-800">
                                 <RefreshCw size={12} />
                             </button>
                         </div>
                     </div>
                     <div className="flex-1 relative bg-white">
                         <iframe 
                            key={previewKey}
                            title="Preview"
                            className="w-full h-full border-none"
                            srcDoc={generatePreviewHtml()}
                            sandbox="allow-scripts"
                         />
                     </div>
                 </div>
             )}
         </div>
      </div>
    </div>
  );
};