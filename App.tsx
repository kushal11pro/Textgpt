import React, { useState, useEffect } from 'react';
import { Bot, Mic, Image as ImageIcon, Eye, Zap, Menu, X, Key, FileCode } from 'lucide-react';
import { AppMode } from './types';
import { ChatInterface } from './components/ChatInterface';
import { LiveSession } from './components/LiveSession';
import { ImageGenInterface } from './components/ImageGenInterface';
import { VisionInterface } from './components/VisionInterface';
import { FastLab } from './components/FastLab';
import { CodeWorkspace } from './components/CodeWorkspace';
import { ApiKeyModal } from './components/ApiKeyModal';
import { hasValidKey } from './utils/apiKey';

const App: React.FC = () => {
  const [activeMode, setActiveMode] = useState<AppMode>(AppMode.CHAT);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isMobile, setIsMobile] = useState(false);
  const [showKeyModal, setShowKeyModal] = useState(false);
  
  // State to hold the prompt when redirecting from Chat to Code
  const [codePrompt, setCodePrompt] = useState<string>('');

  // Handle screen resize
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (mobile) {
        setIsSidebarOpen(false);
      } else {
        setIsSidebarOpen(true);
      }
    };

    // Initial check
    handleResize();
    
    // Check for API key on mount
    if (!hasValidKey()) {
      setShowKeyModal(true);
    }

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleCodeRedirect = (prompt: string) => {
    setCodePrompt(prompt);
    setActiveMode(AppMode.CODING);
  };

  const renderContent = () => {
    switch (activeMode) {
      case AppMode.CHAT: 
        return <ChatInterface onCodeRequest={handleCodeRedirect} />;
      case AppMode.LIVE: return <LiveSession />;
      case AppMode.IMAGES: return <ImageGenInterface />;
      case AppMode.VISION: return <VisionInterface />;
      case AppMode.FAST: return <FastLab />;
      case AppMode.CODING: 
        return <CodeWorkspace initialPrompt={codePrompt} onClearInitialPrompt={() => setCodePrompt('')} />;
      default: 
        return <ChatInterface onCodeRequest={handleCodeRedirect} />;
    }
  };

  const NavItem = ({ mode, icon: Icon, label }: { mode: AppMode; icon: any; label: string }) => (
    <button
      onClick={() => {
        setActiveMode(mode);
        if (isMobile) setIsSidebarOpen(false);
      }}
      className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
        activeMode === mode 
          ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/20' 
          : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
      }`}
    >
      <Icon size={20} />
      {(isSidebarOpen || isMobile) && <span className="font-medium">{label}</span>}
    </button>
  );

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 font-sans overflow-hidden">
      
      <ApiKeyModal 
        isOpen={showKeyModal} 
        onClose={() => setShowKeyModal(false)}
        onSuccess={() => setShowKeyModal(false)}
      />

      {/* Mobile Overlay */}
      {isMobile && isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/60 z-40 backdrop-blur-sm"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div 
        className={`
          bg-slate-900 border-r border-slate-800 flex flex-col transition-all duration-300 ease-in-out z-50
          ${isMobile ? 'fixed inset-y-0 left-0 h-full w-64 shadow-2xl' : 'relative h-full'}
          ${!isMobile && (isSidebarOpen ? 'w-64' : 'w-20')}
          ${isMobile && !isSidebarOpen ? '-translate-x-full' : 'translate-x-0'}
        `}
      >
        <div className="p-5 flex items-center justify-between border-b border-slate-800 h-16">
          {(isSidebarOpen || isMobile) ? (
             <h1 className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-pink-400 bg-clip-text text-transparent">
               Gemini Suite
             </h1>
          ) : (
            <div className="w-full flex justify-center">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-pink-500" />
            </div>
          )}
          
          {/* Close button on mobile, Toggle on Desktop */}
          {isMobile ? (
            <button onClick={() => setIsSidebarOpen(false)} className="text-slate-400 hover:text-white">
              <X size={24} />
            </button>
          ) : (
            <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="text-slate-500 hover:text-white">
              <Menu size={20} />
            </button>
          )}
        </div>

        <nav className="flex-1 p-3 space-y-2 mt-4 overflow-y-auto">
          <NavItem mode={AppMode.CHAT} icon={Bot} label="Chat & Search" />
          <NavItem mode={AppMode.LIVE} icon={Mic} label="Live Voice" />
          <NavItem mode={AppMode.IMAGES} icon={ImageIcon} label="Image Studio" />
          <NavItem mode={AppMode.VISION} icon={Eye} label="Vision Analysis" />
          <NavItem mode={AppMode.FAST} icon={Zap} label="Fast Lab" />
          <NavItem mode={AppMode.CODING} icon={FileCode} label="Code Studio" />
        </nav>

        <div className="p-3 border-t border-slate-800">
          <button
            onClick={() => setShowKeyModal(true)}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-white transition-all"
          >
            <Key size={20} />
            {(isSidebarOpen || isMobile) && <span className="font-medium">API Key</span>}
          </button>
          
           {(isSidebarOpen || isMobile) && (
             <div className="mt-2 text-xs text-slate-600 text-center">
               Powered by Google Gemini 2.5 & 3.0
             </div>
           )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden relative w-full">
        
        {/* Mobile Header to open menu */}
        {isMobile && (
          <div className="bg-slate-900 border-b border-slate-800 p-4 flex items-center justify-between h-16 shrink-0">
             <div className="flex items-center gap-3">
               <button onClick={() => setIsSidebarOpen(true)} className="text-slate-200">
                 <Menu size={24} />
               </button>
               <span className="font-bold text-white">Gemini Suite</span>
             </div>
          </div>
        )}

        <div className="flex-1 overflow-hidden relative">
          {renderContent()}
        </div>
      </div>
    </div>
  );
};

export default App;