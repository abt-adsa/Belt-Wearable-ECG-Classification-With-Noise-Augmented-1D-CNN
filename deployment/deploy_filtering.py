#!/usr/bin/env python3
"""
Filtering Process - AHA-compliant ECG signal processing pipeline.
1. IIR Notch (61.0 Hz, Q=10.0) - Remove powerline interference
2. FIR Equiripple LPF (150/180 Hz) - Remove EMG noise
3. IIR Elliptic HPF (1/0.05 Hz) - Remove baseline wander
4. Min-Max normalization [0, 1] - ML inference ready
"""

import multiprocessing as mp
import numpy as np
from scipy import signal as sp_signal
from pathlib import Path

ADC_RESOLUTION = 16
ADC_MAX_VALUE = (1 << ADC_RESOLUTION) - 1

FIR_PASSBAND_HZ = 150
FIR_STOPBAND_HZ = 180
FIR_PASSBAND_RIPPLE_DB = 0.5
FIR_STOPBAND_ATTEN_DB = 40

HPF_PASSBAND_HZ = 1.0
HPF_STOPBAND_HZ = 0.05
HPF_PASSBAND_RIPPLE_DB = 0.5
HPF_STOPBAND_ATTEN_DB = 40

NOTCH_FREQ_HZ = 61.0
NOTCH_Q = 10.0

SAMPLE_RATE_HZ = 360

class FilteringProcess:
    """AHA-compliant signal processing pipeline for ECG segments."""
    
    def __init__(self,
                 segment_queue: mp.Queue,
                 inference_queue: mp.Queue,
                 status_queue: mp.Queue,
                 output_dir: str = "deployment_data"):
        self.segment_queue = segment_queue
        self.inference_queue = inference_queue
        self.status_queue = status_queue
        self.output_dir = Path(output_dir)
        self.running = False
        self.processed_count = 0
    
    def apply_fir_lowpass(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply FIR equiripple low-pass filter to remove EMG noise."""
        nyquist = SAMPLE_RATE_HZ / 2
        
        bands = [0, FIR_PASSBAND_HZ, FIR_STOPBAND_HZ, nyquist]
        desired = [1, 0]
        
        passband_weight = 1.0
        stopband_weight = 10 ** (FIR_STOPBAND_ATTEN_DB / 20)
        
        width = (FIR_STOPBAND_HZ - FIR_PASSBAND_HZ) / nyquist
        numtaps = int(np.ceil((FIR_STOPBAND_ATTEN_DB - 7.95) / (2.285 * width * np.pi))) + 1
        if numtaps % 2 == 0:
            numtaps += 1
        
        b = sp_signal.remez(numtaps, bands, desired, fs=SAMPLE_RATE_HZ,
                            weight=[passband_weight, stopband_weight])
        
        filtered = sp_signal.filtfilt(b, 1, signal_data)
        
        return filtered
    
    def apply_iir_highpass(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply IIR elliptic high-pass filter to remove baseline wander."""
        nyquist = SAMPLE_RATE_HZ / 2
        
        wp = HPF_PASSBAND_HZ / nyquist
        ws = HPF_STOPBAND_HZ / nyquist
        
        N, Wn = sp_signal.ellipord(wp, ws, HPF_PASSBAND_RIPPLE_DB, HPF_STOPBAND_ATTEN_DB)
        b, a = sp_signal.ellip(N, HPF_PASSBAND_RIPPLE_DB, HPF_STOPBAND_ATTEN_DB, Wn, btype='high')
        
        filtered = sp_signal.filtfilt(b, a, signal_data)
        
        return filtered
    
    def apply_notch_filter(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply notch filter to remove 60 Hz powerline noise."""
        b, a = sp_signal.iirnotch(NOTCH_FREQ_HZ, NOTCH_Q, SAMPLE_RATE_HZ)
        
        filtered = sp_signal.filtfilt(b, a, signal_data)
        
        return filtered
    
    def normalize_segment(self, signal_data: np.ndarray) -> np.ndarray:
        """Min-Max normalization to [0, 1] range."""
        min_val = np.min(signal_data)
        max_val = np.max(signal_data)
        
        normalized = (signal_data - min_val) / (max_val - min_val + 1e-8)
        
        return normalized
    
    def apply_filtering_pipeline(self, signal: np.ndarray) -> np.ndarray:
        """Apply complete AHA-compliant filtering pipeline to ECG signal."""
        signal_float = signal.astype(np.float32)
        
        filtered = self.apply_notch_filter(signal_float)
        filtered = self.apply_fir_lowpass(filtered)
        filtered = self.apply_iir_highpass(filtered)
        
        return filtered
    
    def process_segment(self, segment: dict) -> dict:
        """Complete AHA-compliant processing pipeline for one segment."""
        segment_id = segment['segment_id']
        extended_timestamps = segment['extended_timestamp_ms']
        extended_adc = segment['extended_adc_value']
        core_timestamps = segment['timestamp_ms']
        core_start_idx = segment['core_start_idx']
        core_end_idx = segment['core_end_idx']
        
        filtered_extended = self.apply_filtering_pipeline(extended_adc)
        
        filtered_core = filtered_extended[core_start_idx:core_end_idx]
        raw_core = extended_adc[core_start_idx:core_end_idx]
        
        normalized_signal = self.normalize_segment(filtered_core)
        
        processed_segment = {
            'segment_id': segment_id,
            'timestamp_ms': core_timestamps,
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'raw_signal': raw_core,
            'filtered_signal': filtered_core,
            'processed_signal': normalized_signal,
            'sample_count': segment['sample_count']
        }
        
        return processed_segment
    
    def filtering_loop(self):
        """Main filtering loop."""
        self.status_queue.put(('INFO', 'Filtering process started'))
        
        while self.running:
            try:
                segment = self.segment_queue.get(timeout=0.1)
                
                processed = self.process_segment(segment)
                
                self.inference_queue.put(processed)
                
                self.processed_count += 1
                
                if self.processed_count % 10 == 0:
                    self.status_queue.put(('DEBUG', 
                        f'Filtering: {self.processed_count} segments processed'))
            
            except mp.queues.Empty:
                continue
            except Exception as e:
                self.status_queue.put(('ERROR', f'Filtering error: {e}'))
    
    def run(self):
        """Process main entry point."""
        self.running = True
        
        try:
            self.filtering_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        self.status_queue.put(('INFO', 
            f'Filtering process terminated. Total processed: {self.processed_count}'))


def filtering_worker(segment_queue: mp.Queue, inference_queue: mp.Queue,
                    status_queue: mp.Queue, control_queue: mp.Queue,
                    output_dir: str = "deployment_data"):
    """Worker function for filtering process."""
    process = FilteringProcess(segment_queue, inference_queue, status_queue, output_dir)
    process.run()
