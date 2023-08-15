"""
Caching utilities for CampusGPT
Response caching for improved performance
"""

import time
import hashlib
from typing import Any, Optional, Dict
from threading import Lock
import pickle


class ResponseCache:
    """Thread-safe response cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize response cache
        
        Args:
            max_size: Maximum number of items in cache
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, key: str) -> str:
        """Generate hash key from input"""
        return hashlib.md5(key.encode('utf-8')).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired"""
        return time.time() - timestamp > self.ttl
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times, key=self.access_times.get)
        self.cache.pop(lru_key, None)
        self.access_times.pop(lru_key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None if not found/expired
        """
        with self.lock:
            cache_key = self._generate_key(key)
            
            if cache_key not in self.cache:
                self.misses += 1
                return None
            
            # Check expiration
            entry = self.cache[cache_key]
            if self._is_expired(entry['timestamp']):
                self.cache.pop(cache_key, None)
                self.access_times.pop(cache_key, None)
                self.misses += 1
                return None
            
            # Update access time
            self.access_times[cache_key] = time.time()
            self.hits += 1
            
            return entry['value']
    
    def set(self, key: str, value: Any) -> None:
        """
        Set item in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            cache_key = self._generate_key(key)
            
            # Evict if at capacity
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Store item
            self.cache[cache_key] = {
                'value': value,
                'timestamp': time.time()
            }
            self.access_times[cache_key] = time.time()
    
    def delete(self, key: str) -> bool:
        """
        Delete item from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if item was deleted, False if not found
        """
        with self.lock:
            cache_key = self._generate_key(key)
            
            if cache_key in self.cache:
                self.cache.pop(cache_key, None)
                self.access_times.pop(cache_key, None)
                return True
            
            return False
    
    def clear(self) -> None:
        """Clear all items from cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.hits = 0
            self.misses = 0
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired items from cache
        
        Returns:
            Number of items removed
        """
        with self.lock:
            expired_keys = []
            current_time = time.time()
            
            for cache_key, entry in self.cache.items():
                if current_time - entry['timestamp'] > self.ttl:
                    expired_keys.append(cache_key)
            
            for cache_key in expired_keys:
                self.cache.pop(cache_key, None)
                self.access_times.pop(cache_key, None)
            
            return len(expired_keys)
    
    @property
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate"""
        total_requests = self.hits + self.misses
        return self.hits / total_requests if total_requests > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': self.size,
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
            'ttl': self.ttl
        }


class DiskCache(ResponseCache):
    """Persistent disk-based cache"""
    
    def __init__(self, cache_dir: str = "./cache", **kwargs):
        """
        Initialize disk cache
        
        Args:
            cache_dir: Directory to store cache files
            **kwargs: Arguments for parent ResponseCache
        """
        super().__init__(**kwargs)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache from disk
        self._load_from_disk()
    
    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key"""
        return self.cache_dir / f"{key}.pkl"
    
    def _load_from_disk(self):
        """Load cache entries from disk"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                
                key = cache_file.stem
                
                # Check if entry is still valid
                if not self._is_expired(entry['timestamp']):
                    self.cache[key] = entry
                    self.access_times[key] = entry['timestamp']
                else:
                    # Remove expired file
                    cache_file.unlink()
                    
            except (pickle.PickleError, FileNotFoundError):
                # Remove corrupted files
                cache_file.unlink(missing_ok=True)
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache and persist to disk"""
        cache_key = self._generate_key(key)
        
        # Call parent method
        super().set(key, value)
        
        # Persist to disk
        try:
            cache_file = self._get_cache_file(cache_key)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache[cache_key], f)
        except (pickle.PickleError, IOError) as e:
            # If we can't persist, remove from memory cache too
            super().delete(key)
            raise e
    
    def delete(self, key: str) -> bool:
        """Delete item from cache and disk"""
        cache_key = self._generate_key(key)
        
        # Delete from memory
        deleted = super().delete(key)
        
        # Delete from disk
        cache_file = self._get_cache_file(cache_key)
        if cache_file.exists():
            cache_file.unlink()
        
        return deleted
    
    def clear(self) -> None:
        """Clear all items from cache and disk"""
        # Clear memory
        super().clear()
        
        # Clear disk files
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


# Import Path here to avoid circular imports
from pathlib import Path