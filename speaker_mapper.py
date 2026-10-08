import json
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Union, Any

from subtitle_parser import SubtitleCue

@dataclass
class SpeakerSegment:
    """Represents a diarized speaker segment."""
    speaker_id: str
    start: float
    end: float
    confidence: float = 1.0


def compute_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    """
    Compute intersection duration between two temporal intervals.
    """
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0.0, overlap_end - overlap_start)


def map_subtitles_to_speakers(
    cues: List[SubtitleCue], 
    diarization_segments: List[Union[dict, SpeakerSegment]], 
    min_overlap_ratio: float = 0.3
) -> List[SubtitleCue]:
    """
    Map subtitle cues to diarized speakers based on temporal overlap.
    
    For each subtitle cue, find which diarized speaker has the most temporal overlap.
    Overlap = intersection(cue_interval, speaker_interval) / cue_duration
    
    Args:
        cues: List of SubtitleCue objects to map.
        diarization_segments: List of diarized segments (SpeakerSegment or dicts).
        min_overlap_ratio: Minimum required overlap ratio to not be flagged.
        
    Returns:
        List of SubtitleCue objects updated with speaker_id, confidence, and flagged fields.
    """
    # Normalize diarization segments
    segments = []
    for seg in diarization_segments:
        if isinstance(seg, dict):
            segments.append(SpeakerSegment(
                speaker_id=seg['speaker_id'],
                start=seg['start'],
                end=seg['end'],
                confidence=seg.get('confidence', 1.0)
            ))
        else:
            segments.append(seg)
            
    for cue in cues:
        cue_duration = cue.end - cue.start
        if cue_duration <= 0:
            cue.speaker_id = None
            cue.confidence = 0.0
            cue.flagged = True
            cue.flag_reason = 'invalid_duration'
            continue
            
        overlaps: Dict[str, float] = defaultdict(float)
        for seg in segments:
            overlap = compute_overlap(cue.start, cue.end, seg.start, seg.end)
            if overlap > 0:
                overlaps[seg.speaker_id] += overlap
                
        if not overlaps:
            cue.speaker_id = None
            cue.confidence = 0.0
            cue.flagged = True
            cue.flag_reason = 'no_speaker_overlap'
            continue
            
        # Sort speakers by highest overlap duration
        sorted_speakers = sorted(overlaps.items(), key=lambda x: x[1], reverse=True)
        top_speaker, top_overlap = sorted_speakers[0]
        overlap_ratio = top_overlap / cue_duration
        
        cue.speaker_id = top_speaker
        cue.confidence = overlap_ratio
        
        if overlap_ratio < min_overlap_ratio:
            cue.flagged = True
            cue.flag_reason = 'low_overlap'
        elif len(sorted_speakers) > 1:
            second_speaker, second_overlap = sorted_speakers[1]
            # Flag as ambiguous if the second speaker is very close to the first (within 10% of cue duration)
            if (top_overlap - second_overlap) / cue_duration < 0.1:
                cue.flagged = True
                cue.flag_reason = 'ambiguous_speaker'
            else:
                cue.flagged = False
                cue.flag_reason = None
        else:
            cue.flagged = False
            cue.flag_reason = None
            
    return cues


def export_speaker_map(cues: List[SubtitleCue], output_path: Union[str, Path]) -> Path:
    """
    Export speaker map as a JSON array.
    
    Fields: subtitle_id, start, end, text, speaker, confidence, flagged, flag_reason
    """
    out_path = Path(output_path)
    data = []
    
    for cue in cues:
        data.append({
            'subtitle_id': getattr(cue, 'id', None) or getattr(cue, 'index', None),
            'start': cue.start,
            'end': cue.end,
            'text': cue.text,
            'speaker': getattr(cue, 'speaker_id', None),
            'confidence': getattr(cue, 'confidence', 0.0),
            'flagged': getattr(cue, 'flagged', False),
            'flag_reason': getattr(cue, 'flag_reason', None)
        })
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    return out_path


def load_speaker_map(path: Union[str, Path], cues: List[SubtitleCue]) -> List[SubtitleCue]:
    """
    Load a previously saved speaker map from JSON and apply speaker assignments to cues.
    Used for manual corrections.
    """
    in_path = Path(path)
    with in_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
        
    for cue in cues:
        cue_id = getattr(cue, 'id', None) or getattr(cue, 'index', None)
        
        # Try to match by ID first
        match = next((item for item in data if item.get('subtitle_id') == cue_id), None)
        
        # Fallback to matching by start/end times if ID is unavailable or missing
        if match is None:
            match = next((item for item in data if abs(item.get('start', 0) - cue.start) < 0.01 
                          and abs(item.get('end', 0) - cue.end) < 0.01), None)
            
        if match:
            cue.speaker_id = match.get('speaker')
            cue.confidence = match.get('confidence', 0.0)
            cue.flagged = match.get('flagged', False)
            cue.flag_reason = match.get('flag_reason')
            
    return cues


def merge_speakers(cues: List[SubtitleCue], old_id: str, new_id: str) -> List[SubtitleCue]:
    """
    Merge two speaker IDs by replacing old_id with new_id in all cues.
    """
    for cue in cues:
        if getattr(cue, 'speaker_id', None) == old_id:
            cue.speaker_id = new_id
    return cues


def get_speaker_stats(cues: List[SubtitleCue]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate and return per-speaker statistics.
    
    Returns a dict mapping speaker_id to:
        - count: number of cues assigned
        - total_duration: sum of cue durations
        - average_confidence: mean overlap confidence
    """
    stats = defaultdict(lambda: {'count': 0, 'total_duration': 0.0, 'sum_confidence': 0.0})
    
    for cue in cues:
        speaker = getattr(cue, 'speaker_id', None)
        if speaker is not None:
            stats[speaker]['count'] += 1
            stats[speaker]['total_duration'] += (cue.end - cue.start)
            stats[speaker]['sum_confidence'] += getattr(cue, 'confidence', 0.0)
            
    result = {}
    for speaker, data in stats.items():
        result[speaker] = {
            'count': data['count'],
            'total_duration': round(data['total_duration'], 3),
            'average_confidence': round(data['sum_confidence'] / data['count'], 4) if data['count'] > 0 else 0.0
        }
        
    return result
