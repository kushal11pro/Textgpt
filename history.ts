export const saveToHistory = <T>(key: string, data: T) => {
  try {
    localStorage.setItem(key, JSON.stringify(data));
  } catch (e) {
    console.warn("Failed to save to history (likely storage limit reached):", e);
  }
};

export const loadFromHistory = <T>(key: string, defaultValue: T): T => {
  try {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : defaultValue;
  } catch (e) {
    return defaultValue;
  }
};

export const clearHistory = (key: string) => {
  localStorage.removeItem(key);
};
