import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Search, Loader2, MapPin } from 'lucide-react';

interface SearchResult {
  name: string;
  full_name: string;
  latitude: number;
  longitude: number;
  type?: string;
}

interface SearchBarProps {
  onSelect: (lat: number, lon: number, displayName?: string) => void;
}

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export const SearchBar: React.FC<SearchBarProps> = ({ onSelect }) => {
  const [query, setQuery] = useState<string>('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isOpen, setIsOpen] = useState<boolean>(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Debounced API fetch with 500ms delay for Nominatim rate limits
  useEffect(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setResults([]);
      setIsLoading(false);
      setIsOpen(false);
      return;
    }

    setIsLoading(true);

    debounceTimerRef.current = setTimeout(async () => {
      try {
        const url = `${API_BASE}/api/search?q=${encodeURIComponent(trimmed)}`;
        const res = await axios.get<SearchResult[]>(url);
        const searchResults = res.data || [];
        setResults(searchResults);
        setIsOpen(searchResults.length > 0);
      } catch (err) {
        console.error('Nominatim place search failed:', err);
        setResults([]);
      } finally {
        setIsLoading(false);
      }
    }, 500);

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [query]);

  // Handle clicking outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Escape') {
      setIsOpen(false);
    }
  };

  const handleSelectSuggestion = (item: SearchResult) => {
    const displayName = item.full_name || item.name;
    onSelect(item.latitude, item.longitude, displayName);
    setQuery(item.name);
    setIsOpen(false);
    setResults([]);
  };

  const getFullNameRemainder = (name: string, fullName: string) => {
    if (!fullName) return '';
    if (fullName.toLowerCase().startsWith(name.toLowerCase())) {
      return fullName.slice(name.length).replace(/^[\s,]+/, '').trim();
    }
    const parts = fullName.split(',');
    if (parts.length > 1) {
      return parts.slice(1).join(',').trim();
    }
    return '';
  };

  return (
    <div ref={containerRef} className="relative w-full max-w-md">
      <div className="relative">
        <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
        <input
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setIsOpen(true);
          }}
          onFocus={() => {
            if (results.length > 0) setIsOpen(true);
          }}
          onKeyDown={handleKeyDown}
          placeholder="Search places, schools, colleges, landmarks..."
          className="w-full bg-slate-950 border border-slate-800 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 focus:outline-none rounded-xl pl-10 pr-10 py-2.5 text-sm text-slate-100 placeholder-slate-500 transition-all shadow-inner"
        />

        {/* Loading Spinner */}
        {isLoading && (
          <div className="absolute right-3.5 top-1/2 -translate-y-1/2">
            <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
          </div>
        )}
      </div>

      {/* Autocomplete Dropdown */}
      {isOpen && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl z-[500] overflow-hidden divide-y divide-slate-800/80 backdrop-blur-xl animate-in fade-in-0 duration-150 max-h-80 overflow-y-auto">
          {results.map((item, idx) => {
            const remainder = getFullNameRemainder(item.name, item.full_name);

            return (
              <button
                key={`${item.name}-${idx}`}
                type="button"
                onClick={() => handleSelectSuggestion(item)}
                className="w-full text-left px-4 py-3 text-xs text-slate-200 hover:bg-slate-800 hover:text-white flex items-center justify-between gap-3 transition-colors focus:bg-slate-800 focus:outline-none"
              >
                <div className="flex items-center gap-2.5 min-w-0 flex-1">
                  <MapPin className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                  <div className="truncate">
                    <span className="font-semibold text-slate-100">
                      {item.name}
                    </span>
                    {remainder && (
                      <span className="text-slate-400 ml-1">
                        — {remainder}
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2 flex-shrink-0">
                  {item.type && (
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 uppercase tracking-wide">
                      {item.type}
                    </span>
                  )}
                  <span className="text-[10px] text-slate-500 font-mono whitespace-nowrap">
                    {item.latitude.toFixed(2)}°, {item.longitude.toFixed(2)}°
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};
