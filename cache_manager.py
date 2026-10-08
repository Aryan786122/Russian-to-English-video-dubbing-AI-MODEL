import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import config

CACHE_BASE_DIR = Path(config.TEMP_DIR) / 'cache'

STAGE_DEPENDENCIES = [
    'diarization',
    'speaker_map',
    'speaker_refs',
    'tts_segments',
    'background_separation',
    'mix'
]

@dataclass
class CacheKey:
    """Cache key containing hashes to identify a unique pipeline run."""
    video_hash: str
    subtitle_hash: str
    settings_hash: str = ''
    
    def get_combined_hash(self) -> str:
        """Combine hashes into a single SHA256 string."""
        h = hashlib.sha256()
        h.update(self.video_hash.encode('utf-8'))
        h.update(self.subtitle_hash.encode('utf-8'))
        h.update(self.settings_hash.encode('utf-8'))
        return h.hexdigest()


def compute_file_hash(path: str | Path) -> str:
    """
    Compute SHA256 hash of a file.
    For files >100MB, hashes only the first 10MB, the last 10MB, and the file size for speed.
    """
    path = Path(path)
    file_size = path.stat().st_size
    h = hashlib.sha256()
    
    if file_size > 100 * 1024 * 1024:
        # Large file: first 10MB + last 10MB + size
        h.update(str(file_size).encode('utf-8'))
        with open(path, 'rb') as f:
            h.update(f.read(10 * 1024 * 1024))
            if file_size > 20 * 1024 * 1024:
                f.seek(-10 * 1024 * 1024, os.SEEK_END)
                h.update(f.read(10 * 1024 * 1024))
    else:
        # Full file hash
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
                
    return h.hexdigest()


def get_cache_dir(cache_key: CacheKey) -> Path:
    """
    Get the cache directory for a given cache key.
    Creates the directory if it doesn't exist.
    """
    combined_hash = cache_key.get_combined_hash()
    cache_dir = CACHE_BASE_DIR / combined_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_stage_base_path(cache_dir: Path, stage: str) -> Path:
    """Get the base path for a stage without extension."""
    return cache_dir / stage


def save_stage_result(cache_key: CacheKey, stage: str, data: Any) -> Path:
    """
    Save the result of a processing stage to cache.
    Saves dicts/lists as JSON. Copies file/directory paths.
    """
    cache_dir = get_cache_dir(cache_key)
    stage_path = _get_stage_base_path(cache_dir, stage)
    
    if isinstance(data, (dict, list)):
        json_path = stage_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return json_path
        
    elif isinstance(data, (str, Path)):
        data_path = Path(data)
        if data_path.is_file():
            target_path = stage_path.with_suffix(data_path.suffix)
            shutil.copy2(data_path, target_path)
            return target_path
        elif data_path.is_dir():
            if stage_path.exists():
                shutil.rmtree(stage_path)
            shutil.copytree(data_path, stage_path)
            return stage_path
            
    raise ValueError(f"Unsupported data type for caching: {type(data)}")


def load_stage_result(cache_key: CacheKey, stage: str) -> Optional[Any]:
    """
    Load cached data for a stage.
    Returns the parsed JSON, or the Path object for files/directories.
    Returns None if there is a cache miss.
    """
    cache_dir = get_cache_dir(cache_key)
    stage_path = _get_stage_base_path(cache_dir, stage)
    
    json_path = stage_path.with_suffix('.json')
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    if stage_path.is_dir():
        return stage_path
        
    # Search for files with the stage name as stem
    if cache_dir.exists():
        for f in cache_dir.iterdir():
            if f.is_file() and f.stem == stage:
                return f
                
    return None


def is_stage_cached(cache_key: CacheKey, stage: str) -> bool:
    """Check if a stage result is successfully cached."""
    return load_stage_result(cache_key, stage) is not None


def invalidate_stage(cache_key: CacheKey, stage: str):
    """
    Remove cached data for a specific stage.
    Cascades to invalidate downstream stages as well.
    """
    if stage not in STAGE_DEPENDENCIES:
        return
        
    start_idx = STAGE_DEPENDENCIES.index(stage)
    stages_to_invalidate = STAGE_DEPENDENCIES[start_idx:]
    
    cache_dir = get_cache_dir(cache_key)
    if not cache_dir.exists():
        return
        
    for s in stages_to_invalidate:
        stage_path = _get_stage_base_path(cache_dir, s)
        json_path = stage_path.with_suffix('.json')
        
        if json_path.exists():
            json_path.unlink()
            
        if stage_path.exists():
            if stage_path.is_dir():
                shutil.rmtree(stage_path)
            else:
                stage_path.unlink()
                
        # Also clean up any files with matching stem (audio files etc)
        for f in cache_dir.iterdir():
            if f.is_file() and f.stem == s:
                f.unlink()


def invalidate_subtitle_cue(cache_key: CacheKey, cue_id: int):
    """
    Invalidate only the TTS cache for a specific subtitle cue.
    Used when a user edits just one subtitle. Also cascades downstream invalidations.
    """
    cache_dir = get_cache_dir(cache_key)
    tts_dir = _get_stage_base_path(cache_dir, 'tts_segments')
    
    # Invalidate specific cue in tts_segments if it exists
    if tts_dir.is_dir():
        for f in tts_dir.iterdir():
            if f.is_file() and f.stem == str(cue_id):
                f.unlink()
                
    # Invalidate downstream stages that depend on the full mix (from background_separation onward)
    if 'tts_segments' in STAGE_DEPENDENCIES:
        start_idx = STAGE_DEPENDENCIES.index('tts_segments') + 1
        stages_to_invalidate = STAGE_DEPENDENCIES[start_idx:]
        
        for s in stages_to_invalidate:
            stage_path = _get_stage_base_path(cache_dir, s)
            json_path = stage_path.with_suffix('.json')
            
            if json_path.exists():
                json_path.unlink()
                
            if stage_path.exists():
                if stage_path.is_dir():
                    shutil.rmtree(stage_path)
                else:
                    stage_path.unlink()
                    
            if cache_dir.exists():
                for f in cache_dir.iterdir():
                    if f.is_file() and f.stem == s:
                        f.unlink()
