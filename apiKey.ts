export const STORAGE_KEY = 'gemini_api_key';

export const getApiKey = (): string => {
  return localStorage.getItem(STORAGE_KEY) || process.env.API_KEY || '';
};

export const setApiKey = (key: string) => {
  localStorage.setItem(STORAGE_KEY, key);
};

export const removeApiKey = () => {
  localStorage.removeItem(STORAGE_KEY);
};

export const hasValidKey = (): boolean => {
  return !!getApiKey();
};