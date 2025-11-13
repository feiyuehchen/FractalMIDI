#!/usr/bin/env python3
"""
MIDI Quality Diagnostic Tool

Analyzes generated MIDI files to check for common issues.

Usage:
    python diagnose_midi.py path/to/file.mid
    python diagnose_midi.py outputs/*/unconditional_*.mid  # Multiple files
"""

import symusic
import numpy as np
import sys
from pathlib import Path
import glob


def diagnose_midi(midi_path):
    """Diagnose a single MIDI file."""
    try:
        score = symusic.Score(midi_path)
    except Exception as e:
        print(f"❌ Failed to load {midi_path}: {e}")
        return None
    
    print(f"\n{'='*70}")
    print(f"📁 File: {Path(midi_path).name}")
    print(f"{'='*70}")
    
    results = {}
    
    for track_idx, track in enumerate(score.tracks):
        if len(track.notes) == 0:
            print(f"\n⚠️  Track {track_idx}: Empty")
            continue
        
        notes = track.notes
        pitches = [n.pitch for n in notes]
        velocities = [n.velocity for n in notes]
        durations = [n.duration for n in notes]
        
        # Estimate time steps (assuming 120 ticks per 16th note)
        max_time = max(n.time + n.duration for n in notes)
        estimated_steps = max_time / 120
        
        # Calculate statistics
        notes_per_step = len(notes) / max(estimated_steps, 1)
        
        print(f"\n📊 Track {track_idx} Statistics:")
        print(f"  Total notes:      {len(notes)}")
        print(f"  Estimated steps:  {estimated_steps:.0f}")
        print(f"  Notes/step:       {notes_per_step:.2f}")
        print(f"  Pitch range:      [{min(pitches)}, {max(pitches)}]")
        print(f"  Velocity range:   [{min(velocities)}, {max(velocities)}]")
        print(f"  Velocity mean:    {np.mean(velocities):.1f}")
        print(f"  Velocity std:     {np.std(velocities):.1f}")
        print(f"  Duration mean:    {np.mean(durations):.0f} ticks")
        
        print(f"\n✓ Quality Assessment:")
        
        issues = []
        warnings = []
        
        # Check note density
        if notes_per_step < 0.5:
            warnings.append(f"Too sparse: {notes_per_step:.1f} notes/step (expected 1-5)")
        elif 0.5 <= notes_per_step <= 10:
            print(f"  ✓ Note density OK: {notes_per_step:.1f} notes/step")
        elif 10 < notes_per_step <= 20:
            warnings.append(f"Somewhat dense: {notes_per_step:.1f} notes/step (expected 1-5)")
        else:
            issues.append(f"Too dense: {notes_per_step:.1f} notes/step (expected 1-5)")
        
        # Check pitch range
        if min(pitches) == 0 and max(pitches) == 127:
            issues.append(f"Using full pitch range [0, 127] (unusual for piano)")
        elif 15 <= min(pitches) <= 45 and 65 <= max(pitches) <= 115:
            print(f"  ✓ Pitch range OK: [{min(pitches)}, {max(pitches)}]")
        else:
            warnings.append(f"Unusual pitch range: [{min(pitches)}, {max(pitches)}] (expected ~[21, 108])")
        
        # Check velocity
        vel_mean = np.mean(velocities)
        vel_std = np.std(velocities)
        if 25 <= vel_mean <= 100 and vel_std > 5:
            print(f"  ✓ Velocity OK: mean={vel_mean:.1f}, std={vel_std:.1f}")
        elif vel_mean < 25:
            warnings.append(f"Velocity too low: mean={vel_mean:.1f}")
        elif vel_mean > 100:
            warnings.append(f"Velocity too high: mean={vel_mean:.1f}")
        elif vel_std <= 5:
            warnings.append(f"Velocity lacks variation: std={vel_std:.1f}")
        
        # Check duration
        dur_mean = np.mean(durations)
        if 100 <= dur_mean <= 600:
            print(f"  ✓ Duration OK: mean={dur_mean:.0f} ticks")
        else:
            warnings.append(f"Unusual duration: mean={dur_mean:.0f} ticks")
        
        # Print issues
        for issue in issues:
            print(f"  ❌ {issue}")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
        
        if not issues and not warnings:
            print(f"  ✓ No issues detected!")
        
        results[track_idx] = {
            'notes': len(notes),
            'notes_per_step': notes_per_step,
            'issues': len(issues),
            'warnings': len(warnings)
        }
    
    return results


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    # Support glob patterns
    patterns = sys.argv[1:]
    midi_files = []
    for pattern in patterns:
        midi_files.extend(glob.glob(pattern))
    
    if not midi_files:
        print(f"❌ No MIDI files found matching: {patterns}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"🔍 MIDI Quality Diagnostic Tool")
    print(f"{'='*70}")
    print(f"Analyzing {len(midi_files)} file(s)...")
    
    all_results = {}
    for midi_file in sorted(midi_files):
        result = diagnose_midi(midi_file)
        if result:
            all_results[midi_file] = result
    
    # Summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"📈 Summary")
        print(f"{'='*70}")
        
        total_issues = sum(
            sum(track['issues'] for track in file_results.values())
            for file_results in all_results.values()
        )
        total_warnings = sum(
            sum(track['warnings'] for track in file_results.values())
            for file_results in all_results.values()
        )
        
        print(f"  Files analyzed:  {len(all_results)}")
        print(f"  Total issues:    {total_issues}")
        print(f"  Total warnings:  {total_warnings}")
        
        if total_issues == 0 and total_warnings == 0:
            print(f"\n  ✅ All files look good!")
        elif total_issues == 0:
            print(f"\n  ⚠️  Some warnings found, but no critical issues")
        else:
            print(f"\n  ❌ Critical issues found - see details above")
            print(f"\n  💡 Suggestions:")
            print(f"     - Try: bash run_inference_tuned.sh")
            print(f"     - Or consider retraining with new code")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()

