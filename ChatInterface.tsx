import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenAI } from '@google/genai';
import { Send, Globe, Loader2, Bot, Paperclip, X, Download, FileCode, Eye, Trash2 } from 'lucide-react';
import { ChatMessage } from '../types';
import { saveToHistory, loadFromHistory, clearHistory } from '../utils/history';

interface ChatInterfaceProps {
  onCodeRequest: (prompt: string) => void;
}

const HISTORY_KEY = 'textgpt_chat_history';

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ onCodeRequest }) => {
  // Initialize from history
  const [messages, setMessages] = useState<ChatMessage[]>(() => 
    loadFromHistory<ChatMessage[]>(HISTORY_KEY, [])
  );
  
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [useSearch, setUseSearch] = useState(true);
  
  // File Upload State
  const [attachment, setAttachment] = useState<{ data: string; mimeType: string; url: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Preview State
  const [previewHtml, setPreviewHtml] = useState<string | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);

  // Save to history whenever messages change
  useEffect(() => {
    saveToHistory(HISTORY_KEY, messages);
  }, [messages]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleClearHistory = () => {
    if (window.confirm("Are you sure you want to clear your chat history?")) {
      setMessages([]);
      clearHistory(HISTORY_KEY);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        // Extract pure base64 for API
        const base64Data = result.split(',')[1]; 
        setAttachment({
          data: base64Data,
          mimeType: file.type,
          url: result
        });
      };
      reader.readAsDataURL(file);
    }
    // Reset input so same file can be selected again if needed
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const removeAttachment = () => {
    setAttachment(null);
  };

  const downloadChat = () => {
    const content = messages.map(m => {
      const time = new Date(m.timestamp).toLocaleTimeString();
      return `[${time}] ${m.role.toUpperCase()}:\n${m.text}\n`;
    }).join('\n-------------------\n\n');
    
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `textgpt-chat-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleSend = async () => {
    if ((!input.trim() && !attachment) || isLoading) return;

    // Intelligent Redirection for Coding Tasks
    const isCodingRequest = /\b(make|create|build|generate|code)\b.*\b(web|page|site|app|game|html|script|component|gym)\b/i.test(input);
    if (isCodingRequest && !attachment) { 
        onCodeRequest(input);
        setInput('');
        return;
    }

    const userMsg: ChatMessage = {
      role: 'user',
      text: input,
      attachment: attachment ? { ...attachment } : undefined,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setAttachment(null); 
    setIsLoading(true);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const modelName = useSearch ? 'gemini-3-flash-preview' : 'gemini-3-pro-preview';
      
      const config: any = {};
      if (useSearch) {
        config.tools = [{ googleSearch: {} }];
      }

      const chat = ai.chats.create({
        model: modelName,
        config: config
      });
      
      let response;

      if (userMsg.attachment) {
        response = await chat.sendMessage({
          message: [
            { text: userMsg.text },
            { inlineData: { mimeType: userMsg.attachment.mimeType, data: userMsg.attachment.data } }
          ]
        });
      } else {
        response = await chat.sendMessage({ message: userMsg.text });
      }
      
      const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
      const sources: { title?: string; uri: string }[] = [];
      
      if (groundingChunks) {
        groundingChunks.forEach((chunk: any) => {
          if (chunk.web) {
            sources.push({ title: chunk.web.title, uri: chunk.web.uri });
          }
        });
      }

      const modelMsg: ChatMessage = {
        role: 'model',
        text: response.text || "I couldn't generate a text response.",
        sources: sources.length > 0 ? sources : undefined,
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, modelMsg]);

    } catch (error) {
      console.error(error);
      setMessages((prev) => [...prev, {
        role: 'model',
        text: "Sorry, I encountered an error. Please try again.",
        timestamp: Date.now()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const extractHtml = (text: string) => {
    const match = text.match(/```html\s*([\s\S]*?)\s*```/);
    return match ? match[1] : null;
  };

  return (
    <div className="flex flex-col h-full bg-slate-900 text-slate-100 relative">
      {/* Header */}
      <div className="p-4 border-b border-slate-700 flex justify-between items-center bg-slate-800">
        <h2 className="text-xl font-bold flex items-center gap-2">
          <Bot className="text-indigo-400" />
          TextGpt
        </h2>
        <div className="flex items-center gap-3">
           {messages.length > 0 && (
             <button
               onClick={handleClearHistory}
               title="Clear Chat History"
               className="p-2 hover:bg-red-500/20 hover:text-red-400 rounded-full text-slate-400 transition-colors"
             >
               <Trash2 size={18} />
             </button>
           )}
          <button
            onClick={downloadChat}
            title="Download Chat History"
            className="p-2 hover:bg-slate-700 rounded-full text-slate-400 hover:text-white transition-colors"
          >
            <Download size={18} />
          </button>
          <button
            onClick={() => setUseSearch(!useSearch)}
            className={`px-3 py-1 rounded-full text-xs font-medium flex items-center gap-2 transition-colors ${
              useSearch ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-400'
            }`}
          >
            <Globe size={14} />
            {useSearch ? 'Search Active' : 'Search Off'}
          </button>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="text-center text-slate-500 mt-20">
            <Bot size={48} className="mx-auto mb-4 opacity-50" />
            <p>Ask me anything. Attach images or toggle search.</p>
          </div>
        )}
        {messages.map((msg, idx) => {
          const htmlContent = msg.role === 'model' ? extractHtml(msg.text) : null;
          return (
            <div
              key={idx}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[85%] rounded-2xl p-4 ${
                  msg.role === 'user'
                    ? 'bg-indigo-600 text-white rounded-br-none'
                    : 'bg-slate-800 text-slate-200 rounded-bl-none border border-slate-700'
                }`}
              >
                {/* Attachment Display */}
                {msg.attachment && (
                  <div className="mb-3">
                    <img 
                      src={msg.attachment.url} 
                      alt="Attachment" 
                      className="max-h-64 rounded-lg border border-white/20"
                    />
                  </div>
                )}
                
                <div className="whitespace-pre-wrap">{msg.text}</div>
                
                {/* HTML Preview Button */}
                {htmlContent && (
                  <div className="mt-3 pt-3 border-t border-slate-600/50 flex justify-end">
                    <button
                      onClick={() => setPreviewHtml(htmlContent)}
                      className="flex items-center gap-2 bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-400 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors border border-emerald-500/30"
                    >
                      <FileCode size={16} />
                      Preview Page
                    </button>
                  </div>
                )}

                {/* Sources Display */}
                {msg.sources && (
                  <div className="mt-3 pt-3 border-t border-slate-600/50">
                    <p className="text-xs text-slate-400 font-semibold mb-1">Sources:</p>
                    <div className="flex flex-wrap gap-2">
                      {msg.sources.map((source, sIdx) => (
                        <a
                          key={sIdx}
                          href={source.uri}
                          target="_blank"
                          rel="noreferrer"
                          className="text-xs bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded text-blue-300 truncate max-w-[200px]"
                        >
                          {source.title || new URL(source.uri).hostname}
                        </a>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-slate-800 p-4 rounded-2xl rounded-bl-none flex items-center gap-2">
              <Loader2 className="animate-spin text-indigo-400" size={18} />
              <span className="text-slate-400 text-sm">Thinking...</span>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 bg-slate-800 border-t border-slate-700">
        {/* Attachment Preview in Input */}
        {attachment && (
          <div className="mb-3 flex items-center gap-2 bg-slate-900 w-fit p-2 rounded-lg border border-slate-700 relative group">
             <img src={attachment.url} alt="Preview" className="h-16 w-16 object-cover rounded" />
             <div className="flex flex-col">
               <span className="text-xs text-slate-400 max-w-[150px] truncate">Image Attached</span>
               <span className="text-[10px] text-slate-500">{attachment.mimeType}</span>
             </div>
             <button 
               onClick={removeAttachment}
               className="absolute -top-2 -right-2 bg-red-500 hover:bg-red-600 text-white rounded-full p-1 shadow-md transition-colors"
             >
               <X size={12} />
             </button>
          </div>
        )}
        
        <div className="flex gap-2 items-end">
          <input 
            type="file" 
            ref={fileInputRef}
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="p-3 text-slate-400 hover:text-white bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
            title="Attach Image"
          >
            <Paperclip size={20} />
          </button>
          
          <div className="flex-1 bg-slate-900 border border-slate-700 rounded-lg flex items-center px-4 focus-within:ring-2 focus-within:ring-indigo-500">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Type your message..."
              className="w-full bg-transparent border-none text-slate-200 py-3 focus:outline-none resize-none max-h-32"
              rows={1}
            />
          </div>
          
          <button
            onClick={handleSend}
            disabled={isLoading || (!input.trim() && !attachment)}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-white p-3 rounded-lg transition-colors h-12 w-12 flex items-center justify-center"
          >
            <Send size={20} />
          </button>
        </div>
      </div>

      {/* HTML Preview Modal */}
      {previewHtml && (
        <div className="absolute inset-0 z-50 bg-slate-950/90 backdrop-blur-sm flex items-center justify-center p-6">
          <div className="bg-white w-full h-full max-w-6xl max-h-[90vh] rounded-xl overflow-hidden flex flex-col shadow-2xl">
            <div className="bg-slate-100 p-3 border-b flex justify-between items-center px-4">
              <h3 className="text-slate-800 font-bold flex items-center gap-2">
                <Eye size={18} className="text-indigo-600" />
                Live Preview
              </h3>
              <button 
                onClick={() => setPreviewHtml(null)}
                className="p-2 hover:bg-slate-200 rounded-full text-slate-600 transition-colors"
              >
                <X size={20} />
              </button>
            </div>
            <div className="flex-1 relative">
               <iframe
                 title="Preview"
                 srcDoc={previewHtml}
                 className="w-full h-full border-none bg-white"
                 sandbox="allow-scripts"
               />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};