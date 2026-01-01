
export enum AppMode {
  CHAT = 'CHAT',
  LIVE = 'LIVE',
  IMAGES = 'IMAGES',
  VISION = 'VISION',
  FAST = 'FAST',
  CODING = 'CODING',
  VIDEO = 'VIDEO'
}

export enum ImageResolution {
  RES_1K = '1K',
  RES_2K = '2K',
  RES_4K = '4K'
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
  sources?: Array<{
    title?: string;
    uri: string;
  }>;
  attachment?: {
    data: string; // Base64 string for API
    mimeType: string;
    url: string; // Data URL for display
  };
  timestamp: number;
}

// Live API Types
export interface LiveConfig {
  voiceName: string;
}

export interface CodeFile {
  filename: string;
  content: string;
  language: string;
}
