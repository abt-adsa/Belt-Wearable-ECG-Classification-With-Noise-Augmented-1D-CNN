#!/usr/bin/env python3
"""
Process 3: Filtering + Normalization (Core 2)
==============================================
Per-segment ECG filtering pipeline (filtfilt, zero-phase).

Processing chain applied independently to each 1024-sample segment:
  1. FIR HPF  (721-tap, 0.5 Hz, Hamming)  — baseline wander removal
  2. IIR LPF  (Butterworth ord-4, 45 Hz)   — EMG / HF noise
  3. IIR Notch (61.7 Hz, Q=10)             — powerline interference
  4. Savitzky-Golay (w=15, p=3)            — light smoothing
  5. Z-score normalization                 — ML inference ready

filtfilt is used for zero-phase output (no group delay, no warmup artefacts).
Each segment is filtered independently — no inter-segment state needed.
Note: Inference process handles all CSV file I/O.
"""

import multiprocessing as mp
import numpy as np
from typing import Dict
from scipy import signal as sp_signal
from scipy.signal import savgol_filter
from pathlib import Path

# Sampling rate
SAMPLE_RATE_HZ = 360.0

# Filter parameters (must match refilter_clinical1.py and retroactive_inference.py)
HPF_CUTOFF_HZ = 1.0
HPF_ORDER     = 4

LPF_CUTOFF_HZ = 35.0
LPF_ORDER     = 4

NOTCH_FREQ_HZ = 61.7
NOTCH_Q       = 10.0

SG_WINDOW     = 15
SG_POLYORDER  = 3


def filter_extended(extended: np.ndarray,
                    b_hpf, a_hpf, b_lpf, a_lpf, b_notch, a_notch) -> np.ndarray:
    """
    Apply filtfilt pipeline to an extended window (core + context overlap).
    Returns the full filtered array — caller slices out the core region.
    """
    s = extended.astype(np.float64)
    s = sp_signal.detrend(s)          # remove DC + linear slope → prevents HPF boundary transients
    s = sp_signal.filtfilt(b_hpf,   a_hpf,   s)
    s = sp_signal.filtfilt(b_lpf,   a_lpf,   s)
    s = sp_signal.filtfilt(b_notch, a_notch, s)
    return savgol_filter(s, SG_WINDOW, SG_POLYORDER)


def filter_segment(raw: np.ndarray,
                   b_hpf, a_hpf, b_lpf, a_lpf, b_notch, a_notch) -> tuple:
    """
    Apply the full pipeline to a bare array (no context overlap).
    Prefer filter_extended() when overlap data is available.

    Returns
    -------
    filtered   : np.ndarray  smoothed signal
    normalized : np.ndarray  per-segment z-score of filtered
    """
    filtered = filter_extended(raw, b_hpf, a_hpf, b_lpf, a_lpf, b_notch, a_notch)
    mean = np.mean(filtered)
    std  = np.std(filtered) + 1e-8
    return filtered, (filtered - mean) / std



class FilteringProcess:
    """
    Per-segment ECG filtering process.

    Filter coefficients are built once at startup.  Each 1024-sample
    segment is filtered independently with filtfilt (zero-phase).
    No inter-segment state is maintained.
    """

    def __init__(self,
                 segment_queue: mp.Queue,
                 inference_queue: mp.Queue,
                 status_queue: mp.Queue,
                 output_dir: str = "../data"):
        self.segment_queue   = segment_queue
        self.inference_queue = inference_queue
        self.status_queue    = status_queue
        self.output_dir      = Path(output_dir)
        self.running         = False
        self.processed_count = 0

        # Build filter coefficients once
        self._b_hpf, self._a_hpf = sp_signal.butter(
            HPF_ORDER, HPF_CUTOFF_HZ, btype='high', fs=SAMPLE_RATE_HZ
        )
        self._b_lpf, self._a_lpf     = sp_signal.butter(
            LPF_ORDER, LPF_CUTOFF_HZ, btype='low', fs=SAMPLE_RATE_HZ
        )
        self._b_notch, self._a_notch = sp_signal.iirnotch(
            NOTCH_FREQ_HZ, NOTCH_Q, SAMPLE_RATE_HZ
        )
    

    
    # ── Core processing ────────────────────────────────────────────────────────

    def process_segment(self, segment: Dict) -> Dict:
        """
        Filter one segment and return a processed dict.

        The extended window (core + 256-sample overlap on each side, provided
        by SegmentationProcess) is passed through filtfilt so the filter has
        real context at both boundaries.  Only the core 1024 samples are
        returned; the overlap is discarded after filtering.
        """
        core_start   = segment['core_start_idx']
        core_end     = segment['core_end_idx']
        extended_raw = np.asarray(segment['extended_adc_value'])
        raw_core     = extended_raw[core_start:core_end]

        # Filter the full extended window — overlap removes edge artifacts
        extended_filtered = filter_extended(
            extended_raw,
            self._b_hpf, self._a_hpf, self._b_lpf, self._a_lpf,
            self._b_notch, self._a_notch
        )

        # Extract core and z-score only the core 1024 samples
        filtered_signal = extended_filtered[core_start:core_end]
        mean = np.mean(filtered_signal)
        std  = np.std(filtered_signal) + 1e-8
        normalized_signal = (filtered_signal - mean) / std

        return {
            'segment_id':       segment['segment_id'],
            'session_id':       segment.get('session_id', 0),
            'timestamp_ms':     segment['timestamp_ms'],
            'start_time':       segment['start_time'],
            'end_time':         segment['end_time'],
            'raw_signal':       raw_core,
            'filtered_signal':  filtered_signal,
            'processed_signal': normalized_signal,
            'sample_count':     segment['sample_count'],
        }
    
    def filtering_loop(self):
        """
        Main filtering loop:
        - Receive segments from segmentation
        - Apply AHA-compliant filtering and normalization
        - Forward to inference process
        """
        self.status_queue.put(('INFO', 'Filtering process started'))
        
        while self.running:
            try:
                # Get segment from segmentation (with timeout)
                segment = self.segment_queue.get(timeout=0.1)
                
                # Process segment
                processed = self.process_segment(segment)
                
                # Forward to inference
                self.inference_queue.put(processed)
                
                self.processed_count += 1
                
                # Status update every 10 segments
                if self.processed_count % 10 == 0:
                    self.status_queue.put(('DEBUG', 
                        f'Filtering: {self.processed_count} segments processed'))
            
            except mp.queues.Empty:
                continue
            except Exception as e:
                self.status_queue.put(('ERROR', f'Filtering error: {e}'))
    
    def run(self):
        """Process main entry point"""
        self.running = True
        
        try:
            self.filtering_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.status_queue.put(('INFO', 
            f'Filtering process terminated. Total processed: {self.processed_count}'))


def filtering_worker(segment_queue: mp.Queue, inference_queue: mp.Queue,
                    status_queue: mp.Queue, control_queue: mp.Queue,
                    output_dir: str = "../data"):
    """
    Worker function for filtering process.
    
    Args:
        segment_queue: Input queue from segmentation
        inference_queue: Output queue to inference
        status_queue: Output queue for status messages
        control_queue: Input queue for control commands
        output_dir: Directory for CSV output files
    """
    process = FilteringProcess(segment_queue, inference_queue, status_queue, output_dir)
    process.run()
