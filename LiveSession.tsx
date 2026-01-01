import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Type, FunctionDeclaration, Schema } from '@google/genai';
import { Mic, MicOff, Activity, Image as ImageIcon, X, Download, Loader2, FileCode } from 'lucide-react';
import { decodeAudioData, createPcmBlob, base64ToUint8Array } from '../utils/audioUtils';
import { saveToHistory, loadFromHistory } from '../utils/history';
import { CodeFile } from '../types';

// Constants for Code Workspace Storage (Must match CodeWorkspace.tsx)
const FILES_KEY = 'textgpt_code_files';
const CHAT_KEY = 'textgpt_code_chat';

// Tool Definition for Image Generation
const generateImageTool: FunctionDeclaration = {
  name: 'generate_image',
  description: 'Generate an image based on a text prompt. Use this when the user asks to create, generate, draw, or make an image/picture/photo.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      prompt: {
        type: Type.STRING,
        description: 'The detailed description of the image to generate.'
      }
    },
    required: ['prompt']
  }
};

// Tool Definition for Code Generation
const generateCodeTool: FunctionDeclaration = {
  name: 'generate_code',
  description: 'Generate code for a web application, website, or script. Use this when the user asks to build, code, create an app, or write software.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      description: {
        type: Type.STRING,
        description: 'The detailed description of the application or code functionality requested.'
      }
    },
    required: ['description']
  }
};

export const LiveSession: React.FC = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [status, setStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  
  // Image Generation State within Live Session
  const [liveImage, setLiveImage] = useState<{ url: string; prompt: string } | null>(null);
  const [isProcessingTool, setIsProcessingTool] = useState(false);
  const [toolStatus, setToolStatus] = useState<string>('');

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sessionRef = useRef<Promise<any> | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const inputContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  
  // Audio Playback Queue
  const nextStartTimeRef = useRef<number>(0);
  const scheduledSources = useRef<Set<AudioBufferSourceNode>>(new Set());

  // Animation frame for visualizer
  const animRef = useRef<number>(0);

  const cleanup = () => {
    if (sessionRef.current) {
      sessionRef.current.then(s => s.close());
      sessionRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (inputContextRef.current) {
      inputContextRef.current.close();
      inputContextRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (animRef.current) {
      cancelAnimationFrame(animRef.current);
    }
    scheduledSources.current.forEach(s => s.stop());
    scheduledSources.current.clear();
    setIsConnected(false);
    setStatus('disconnected');
    setIsSpeaking(false);
    setLiveImage(null);
    setIsProcessingTool(false);
  };

  // Function to generate image using standard API
  const generateImageFromVoice = async (prompt: string): Promise<string> => {
    setToolStatus('Generating image...');
    setIsProcessingTool(true);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: { parts: [{ text: prompt }] },
        config: {
            imageConfig: { aspectRatio: '1:1' }
        }
      });

      let imageUrl = '';
      if (response.candidates?.[0]?.content?.parts) {
        for (const part of response.candidates[0].content.parts) {
          if (part.inlineData) {
            imageUrl = `data:image/png;base64,${part.inlineData.data}`;
            break;
          }
        }
      }
      
      if (imageUrl) {
          setLiveImage({ url: imageUrl, prompt });
          return "Image generated and displayed successfully.";
      } else {
          return "Failed to generate image.";
      }
    } catch (e) {
        console.error("Live image gen error", e);
        return "Error occurred while generating image.";
    } finally {
        setIsProcessingTool(false);
    }
  };

  // Function to generate code and save to storage
  const generateCodeFromVoice = async (description: string): Promise<string> => {
    setToolStatus('Writing code...');
    setIsProcessingTool(true);
    try {
        const apiKey = process.env.API_KEY;
        const ai = new GoogleGenAI({ apiKey });
        
        // Use the same schema as CodeWorkspace for compatibility
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

        const existingFiles = loadFromHistory<CodeFile[]>(FILES_KEY, []);
        
        let systemInstruction = "You are an expert full-stack developer. Always output code in a structured JSON format containing a list of files. Ensure code is complete, functional, and production-ready.";
        let prompt = `Create code for: ${description}. Generate multiple files if needed. Return JSON.`;
        
        if (existingFiles.length > 0) {
            prompt += `\n\nCurrent File Structure: ${existingFiles.map(f => f.filename).join(', ')}. \nUpdate or create files as needed.`;
        }

        const response = await ai.models.generateContent({
            model: 'gemini-3-flash-preview',
            contents: { parts: [{ text: prompt }] },
            config: {
                responseMimeType: 'application/json',
                responseSchema: responseSchema,
                systemInstruction: systemInstruction
            }
        });

        const jsonStr = response.text;
        if (!jsonStr) return "Failed to generate code.";

        const result = JSON.parse(jsonStr);
        
        // Update Local Storage Logic
        if (result.files && Array.isArray(result.files)) {
            // Merge files
            const newFilesMap = new Map(existingFiles.map(f => [f.filename, f]));
            result.files.forEach((f: CodeFile) => {
                newFilesMap.set(f.filename, f);
            });
            const updatedFiles = Array.from(newFilesMap.values());
            
            // Save Files
            saveToHistory(FILES_KEY, updatedFiles);
            
            // Save Chat History Context
            const existingChat = loadFromHistory<any[]>(CHAT_KEY, []);
            const updatedChat = [
                ...existingChat, 
                { role: 'user', text: description },
                { role: 'model', text: result.explanation || "Code generated via Voice Mode." }
            ];
            saveToHistory(CHAT_KEY, updatedChat);

            return `Code generated successfully for "${description}". Files saved to Code Studio.`;
        }
        
        return "Code generation failed format check.";

    } catch (e) {
        console.error("Live code gen error", e);
        return "Error occurred while generating code.";
    } finally {
        setIsProcessingTool(false);
    }
  };

  const startSession = async () => {
    try {
      setStatus('connecting');
      
      // Initialize Audio Contexts
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      inputContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      
      nextStartTimeRef.current = audioContextRef.current.currentTime;

      // Get Mic Stream
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      const sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } },
          },
          systemInstruction: "You are a helpful, witty AI assistant. You have access to tools for generating images and writing code/apps. \n\n1. If asked to generate/create/draw an image, call 'generate_image'. \n2. If asked to write code, build an app, or create a website, call 'generate_code'. \n\nInform the user when you are using a tool. For code, tell them to check the 'Code Studio' tab after you finish.",
          tools: [{ functionDeclarations: [generateImageTool, generateCodeTool] }],
        },
        callbacks: {
          onopen: () => {
            console.log('Live session opened');
            setStatus('connected');
            setIsConnected(true);
            
            // Setup Input Processing
            if (!inputContextRef.current) return;
            
            const source = inputContextRef.current.createMediaStreamSource(stream);
            sourceRef.current = source;
            
            // Use ScriptProcessor as per guidelines (standard Web Audio API for raw PCM access)
            const processor = inputContextRef.current.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;
            
            processor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              const pcmBlob = createPcmBlob(inputData);
              
              sessionPromise.then(session => {
                session.sendRealtimeInput({ media: pcmBlob });
              });
            };
            
            source.connect(processor);
            processor.connect(inputContextRef.current.destination);
          },
          onmessage: async (msg: LiveServerMessage) => {
            // Handle Tool Calls (Function Calling)
            if (msg.toolCall) {
                for (const fc of msg.toolCall.functionCalls) {
                    let result = "Tool execution failed";
                    
                    if (fc.name === 'generate_image') {
                        const prompt = fc.args['prompt'] as string;
                        result = await generateImageFromVoice(prompt);
                    } else if (fc.name === 'generate_code') {
                        const desc = fc.args['description'] as string;
                        result = await generateCodeFromVoice(desc);
                    }

                    // Send response back to model
                    sessionPromise.then(session => {
                        session.sendToolResponse({
                            functionResponses: {
                                name: fc.name,
                                id: fc.id,
                                response: { result: result }
                            }
                        });
                    });
                }
            }

            const audioData = msg.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioData && audioContextRef.current) {
              setIsSpeaking(true);
              const ctx = audioContextRef.current;
              
              // Ensure gapless playback
              nextStartTimeRef.current = Math.max(
                nextStartTimeRef.current,
                ctx.currentTime
              );
              
              const audioBuffer = await decodeAudioData(
                base64ToUint8Array(audioData),
                ctx,
                24000,
                1
              );
              
              const source = ctx.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(ctx.destination);
              
              source.addEventListener('ended', () => {
                scheduledSources.current.delete(source);
                if (scheduledSources.current.size === 0) {
                   setIsSpeaking(false);
                }
              });
              
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += audioBuffer.duration;
              scheduledSources.current.add(source);
            }

            if (msg.serverContent?.turnComplete) {
               setIsSpeaking(false);
            }
          },
          onclose: () => {
            console.log('Live session closed');
            cleanup();
          },
          onerror: (err) => {
            console.error('Live session error', err);
            cleanup();
            setStatus('error');
          }
        }
      });
      
      sessionRef.current = sessionPromise;

    } catch (e) {
      console.error("Failed to start live session", e);
      cleanup();
      setStatus('error');
    }
  };

  // Visualizer Effect
  useEffect(() => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    let hue = 0;
    const draw = () => {
      if (!ctx || !canvasRef.current) return;
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      const cx = canvasRef.current.width / 2;
      const cy = canvasRef.current.height / 2;
      
      if (isConnected) {
        hue += 1;
        ctx.strokeStyle = `hsl(${hue % 360}, 70%, 60%)`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        // Simple circle pulse simulation
        const r = 50 + (isSpeaking ? Math.random() * 20 : 0);
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();

        if (isSpeaking) {
          ctx.beginPath();
          ctx.arc(cx, cy, r + 10, 0, Math.PI * 2);
          ctx.strokeStyle = `hsl(${(hue + 180) % 360}, 50%, 50%)`;
          ctx.stroke();
        }
      } else {
        ctx.fillStyle = '#334155';
        ctx.beginPath();
        ctx.arc(cx, cy, 40, 0, Math.PI * 2);
        ctx.fill();
      }
      
      animRef.current = requestAnimationFrame(draw);
    };
    draw();
    return () => {
        if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [isConnected, isSpeaking]);

  return (
    <div className="flex flex-col h-full bg-slate-900 items-center justify-center p-6 relative overflow-hidden">
      {/* Background decorations */}
      <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-indigo-900/20 via-slate-900 to-slate-900 pointer-events-none" />
      
      <div className="z-10 text-center space-y-8 flex flex-col items-center w-full max-w-lg">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2 tracking-tight">TextGpt Live</h2>
          <p className="text-slate-400">Real-time multimodal conversation</p>
        </div>

        {/* Visualizer / Image Container */}
        <div className="relative w-full flex flex-col items-center gap-6">
          <div className="relative">
             <canvas 
               ref={canvasRef} 
               width={300} 
               height={300} 
               className="rounded-full bg-slate-800/50 shadow-2xl backdrop-blur-sm"
             />
             <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
               {isConnected ? (
                 <Activity className={`w-12 h-12 ${isSpeaking ? 'text-indigo-400' : 'text-slate-500'} transition-colors`} />
               ) : (
                 <MicOff className="w-12 h-12 text-slate-600" />
               )}
             </div>
          </div>

          {/* Generated Image Overlay/Card */}
          {liveImage && (
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 md:w-80 bg-slate-800 p-2 rounded-xl shadow-2xl border border-slate-700 animate-in fade-in zoom-in duration-300">
                  <div className="relative">
                      <img src={liveImage.url} alt="Generated" className="w-full h-auto rounded-lg" />
                      <button 
                         onClick={() => setLiveImage(null)}
                         className="absolute -top-3 -right-3 bg-slate-700 hover:bg-red-500 text-white rounded-full p-1 shadow-lg transition-colors border border-slate-600"
                      >
                          <X size={14} />
                      </button>
                      <div className="absolute bottom-2 right-2">
                           <a 
                             href={liveImage.url} 
                             download="textgpt-live-gen.png" 
                             className="p-1.5 bg-black/60 hover:bg-black/80 text-white rounded-md flex backdrop-blur-md transition-colors"
                           >
                              <Download size={14} />
                           </a>
                      </div>
                  </div>
                  <p className="text-xs text-slate-400 mt-2 px-1 truncate">{liveImage.prompt}</p>
              </div>
          )}

           {isProcessingTool && (
              <div className="absolute top-full mt-4 flex items-center gap-2 bg-slate-800/80 px-4 py-2 rounded-full text-indigo-300 text-sm backdrop-blur-sm border border-slate-700/50">
                  <Loader2 className="animate-spin" size={14} />
                  {toolStatus || 'Processing...'}
              </div>
           )}
        </div>

        <div className="flex justify-center gap-4">
          {!isConnected ? (
            <button
              onClick={startSession}
              disabled={status === 'connecting'}
              className="group relative inline-flex items-center justify-center px-8 py-3 text-lg font-medium text-white transition-all duration-200 bg-indigo-600 rounded-full hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {status === 'connecting' ? 'Connecting...' : 'Start Conversation'}
              <Mic className="ml-2 w-5 h-5 group-hover:scale-110 transition-transform" />
            </button>
          ) : (
            <button
              onClick={cleanup}
              className="inline-flex items-center justify-center px-8 py-3 text-lg font-medium text-white transition-all duration-200 bg-red-600 rounded-full hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
            >
              End Session
            </button>
          )}
        </div>
        
        {status === 'error' && (
          <p className="text-red-400 text-sm mt-4">Connection failed. Please check permissions and try again.</p>
        )}
      </div>
    </div>
  );
};