#!/usr/bin/env python3
"""
Process 4: Inference + File I/O (Core 3)
=========================================
I/O-bound process:
- TFLite inference (4 rhythm classes)
- Write signal CSV (timestamp_ms, adc_value)
- Write rhythm annotation file (synchronized with signal)
- Handle file transfers

Runs on dedicated core to prevent I/O blocking inference.
"""

import multiprocessing as mp
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import numpy as np
from scipy.signal import find_peaks, peak_widths
from gpiozero import Buzzer

RHYTHM_CLASSES = {
    0: 'NSR',
    1: 'AFIB',
    2: 'PVC',
    3: 'LBBB'
}

BUZZER_PIN = 6

# Timing constants based on ECG segment duration
SEGMENT_DURATION = 1024 / 360  # 2.844 seconds
UNIT = SEGMENT_DURATION / 8     # Base timing unit (~0.355 seconds)

SAMPLE_RATE_HZ = 360.0
RR_FEATURE_COUNT = 7

# Beep durations (Morse code patterns)
SHORT_BEEP = UNIT * 1      # ~0.355s (dot)
LONG_BEEP = UNIT * 3       # ~1.066s (dash)
VERY_LONG_BEEP = UNIT * 6  # ~2.133s (extra long for LBBB)

# Gaps
ELEMENT_GAP = UNIT * 1     # ~0.355s (between dots/dashes)
PATTERN_GAP = UNIT * 3     # ~1.066s (between pattern repeats)

class InferenceProcess:
    """TFLite inference and file I/O management."""
    
    def __init__(self,
                 inference_queue: mp.Queue,
                 status_queue: mp.Queue,
                 control_queue: mp.Queue,
                 output_dir: Path):
        self.inference_queue = inference_queue
        self.status_queue = status_queue
        self.control_queue = control_queue
        self.output_dir = output_dir
        self.running = False
        
        # File handles
        self.signal_file = None
        self.signal_writer = None
        self.filtered_file = None
        self.filtered_writer = None
        self.annotation_file = None
        self.annotation_writer = None
        self.current_session = None
        self.current_session_id = 0  # Track session to reject stale segments
        
        # Buzzer for abnormal rhythm alerts
        self.buzzer = None
        
        # TFLite interpreter
        self.interpreter = None
        self.input_details = []
        self.output_details = []
        self.input_detail_map = {}
        self.signal_input_index = None
        self.feature_input_index = None
        
        # Statistics
        self.inference_count = 0
        self.samples_written = 0
        
        # Recording duration tracking
        self.recording_start_time = None
        self.recording_duration = 0

        # Last classification — for 1-second OLED heartbeat
        self._last_label      = None
        self._last_heartbeat  = 0.0
    
    def setup(self):
        """Initialize resources"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load TFLite model
        self.interpreter = self.load_tflite_model()
        
        # Initialize buzzer (active_high=False for low-level trigger)
        try:
            self.buzzer = Buzzer(BUZZER_PIN, active_high=False)
            self.status_queue.put(('INFO', f'Buzzer initialized on GPIO {BUZZER_PIN}'))
        except Exception as e:
            self.status_queue.put(('WARN', f'Buzzer init failed: {e}'))
            self.buzzer = None
        
        self.status_queue.put(('INFO', 'Inference process initialized'))
    
    def load_tflite_model(self):
        """
        Load TFLite model for inference.
        
        Returns:
            TFLite Interpreter instance
        """
        runtime_name = None
        try:
            from ai_edge_litert.interpreter import Interpreter
            runtime_name = 'ai_edge_litert'
        except Exception as litert_err:
            self.status_queue.put(('WARN', f'ai_edge_litert import failed: {litert_err}'))
            try:
                from tflite_runtime.interpreter import Interpreter
                runtime_name = 'tflite_runtime'
            except Exception as rt_err:
                self.status_queue.put(('ERROR', f'No LiteRT/TFLite runtime available: {rt_err}'))
                return None

        model_candidates = [
            Path(__file__).parent / 'model.tflite',
            Path(__file__).parent / 'model_zscore.tflite',
            Path(__file__).parent / 'model_minmax.tflite',
        ]
        model_path = next((p for p in model_candidates if p.exists()), None)

        if model_path is None:
            self.status_queue.put(('ERROR', 'No model file found (tried model.tflite, model_zscore.tflite, model_minmax.tflite)'))
            return None

        try:
            interpreter = Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()

            self.input_details = interpreter.get_input_details()
            self.output_details = interpreter.get_output_details()
            self.input_detail_map = {d['index']: d for d in self.input_details}
            self._resolve_input_tensors()

            self.status_queue.put(('INFO', f'Loaded model ({runtime_name}): {model_path.name}'))
            return interpreter
        except Exception as e:
            self.status_queue.put(('ERROR', f'Failed to load model: {e}'))
            return None

    def _resolve_input_tensors(self):
        """Map model inputs to signal and RR-feature tensors by name/shape."""
        self.signal_input_index = None
        self.feature_input_index = None

        for detail in self.input_details:
            tensor_name = str(detail.get('name', '')).lower()
            shape = tuple(int(v) for v in detail.get('shape', ()))

            is_signal = (
                'signal' in tensor_name
                or (len(shape) >= 2 and 1024 in shape)
                or (len(shape) == 3 and shape[-1] == 1)
            )
            is_feature = (
                'feat' in tensor_name
                or 'feature' in tensor_name
                or (len(shape) == 2 and shape[-1] == RR_FEATURE_COUNT)
            )

            if is_signal and self.signal_input_index is None:
                self.signal_input_index = detail['index']
                continue
            if is_feature and self.feature_input_index is None:
                self.feature_input_index = detail['index']

        if self.signal_input_index is None and self.input_details:
            self.signal_input_index = self.input_details[0]['index']

        if self.feature_input_index is None and len(self.input_details) >= 2:
            for detail in self.input_details:
                if detail['index'] != self.signal_input_index:
                    self.feature_input_index = detail['index']
                    break

        if self.feature_input_index is not None:
            self.status_queue.put(('INFO', 'Dual-input model detected (signal + RR features)'))
        else:
            self.status_queue.put(('WARN', 'Single-input model detected; RR feature tensor disabled'))

    @staticmethod
    def _quantize_if_needed(values: np.ndarray, detail: Dict) -> np.ndarray:
        """Cast/quantize input to the dtype expected by the model tensor."""
        dtype = detail['dtype']
        if np.issubdtype(dtype, np.floating):
            return values.astype(dtype, copy=False)

        scale, zero_point = detail.get('quantization', (0.0, 0))
        if scale and scale > 0:
            q = np.round(values / scale + zero_point)
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                q = np.clip(q, info.min, info.max)
            return q.astype(dtype)

        return values.astype(dtype)

    @staticmethod
    def _dequantize_if_needed(values: np.ndarray, detail: Dict) -> np.ndarray:
        """Convert output tensor to float probabilities when model is quantized."""
        arr = np.asarray(values)
        if np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float32, copy=False)

        scale, zero_point = detail.get('quantization', (0.0, 0))
        if scale and scale > 0:
            return (arr.astype(np.float32) - zero_point) * scale

        return arr.astype(np.float32)

    @staticmethod
    def _extract_rr_features(processed_signal: np.ndarray) -> np.ndarray:
        """
        Match training-time RR feature extraction from ecg_classifier_v6_8.py.

        Features order:
        [mean_rr, std_rr, rmssd, pnn50, min_rr, max_rr, mean_qrs]
        """
        sig = np.asarray(processed_signal, dtype=np.float32).flatten()
        if sig.size == 0:
            return np.array([0, 0, 0, 1.0, 0, 0, 0.0], dtype=np.float32)

        min_dist = int(0.2 * SAMPLE_RATE_HZ)
        max_val = float(np.max(sig))

        if max_val < 0.1:
            return np.array([0, 0, 0, 1.0, 0, 0, 0.0], dtype=np.float32)

        peaks, _ = find_peaks(sig, height=max_val * 0.3, distance=min_dist)
        if len(peaks) < 2:
            return np.array([0, 0, 0, 1.0, 0, 0, 0.0], dtype=np.float32)

        rrs = np.diff(peaks) / SAMPLE_RATE_HZ

        mean_rr = np.mean(rrs)
        std_rr = np.std(rrs)
        min_rr = np.min(rrs)
        max_rr = np.max(rrs)

        diff_rrs = np.diff(rrs)
        rmssd = np.sqrt(np.mean(diff_rrs ** 2)) if len(diff_rrs) > 0 else 0.0
        pnn50 = np.sum(np.abs(diff_rrs) > 0.050) / len(diff_rrs) if len(diff_rrs) > 0 else 0.0

        widths, _, _, _ = peak_widths(sig, peaks, rel_height=0.5)
        mean_qrs = np.mean(widths) / SAMPLE_RATE_HZ if len(widths) > 0 else 0.0

        return np.array([mean_rr, std_rr, rmssd, pnn50, min_rr, max_rr, mean_qrs], dtype=np.float32)
    
    def start_session(self, session_name: str):
        """
        Start a new recording session.
        
        Args:
            session_name: Name for this recording session
        """
        self.current_session = session_name
        self.current_session_id += 1  # Increment session ID
        # recording_start_time already set in NEW_SESSION handler — do not reset
        self.recording_duration = 0
        self._last_label     = None   # reset heartbeat for new session
        self._last_heartbeat = 0.0
        # Tell OLED we're buffering data before first classification
        self.status_queue.put(('STATUS', 'BUFFERING'))
        
        # Create signal CSV file (raw ADC values)
        signal_path = self.output_dir / f"{session_name}_signal.csv"
        self.signal_file = open(signal_path, 'w', newline='')
        self.signal_writer = csv.writer(self.signal_file)
        self.signal_writer.writerow(['timestamp_ms', 'adc_value'])
        
        # Create filtered CSV file (AHA-filtered signal)
        filtered_path = self.output_dir / f"{session_name}_filtered.csv"
        self.filtered_file = open(filtered_path, 'w', newline='')
        self.filtered_writer = csv.writer(self.filtered_file)
        self.filtered_writer.writerow(['timestamp_ms', 'adc_value'])
        
        # Create rhythm annotation file
        annotation_path = self.output_dir / f"{session_name}_rhythm.csv"
        self.annotation_file = open(annotation_path, 'w', newline='')
        self.annotation_writer = csv.writer(self.annotation_file)
        self.annotation_writer.writerow(['segment_id', 'start_time_ms', 'end_time_ms', 
                                        'rhythm_class', 'rhythm_label', 'confidence'])
        
        self.status_queue.put(('STATUS', f'SESSION_START:{session_name}'))
        self.status_queue.put(('INFO', f'Signal file: {signal_path}'))
        self.status_queue.put(('INFO', f'Rhythm file: {annotation_path}'))
    
    def stop_session(self):
        """Stop current recording session and close files"""
        # Calculate final duration
        if self.recording_start_time:
            self.recording_duration = int(time.time() - self.recording_start_time)
        
        if self.signal_file:
            self.signal_file.close()
            self.signal_file = None
            self.signal_writer = None
        
        if self.filtered_file:
            self.filtered_file.close()
            self.filtered_file = None
            self.filtered_writer = None
        
        if self.annotation_file:
            self.annotation_file.close()
            self.annotation_file = None
            self.annotation_writer = None
        
        # Show saved filename and duration on OLED
        self.status_queue.put(('STATUS', f'SAVED:{self.current_session}:{self.recording_duration}'))
        
        self.status_queue.put(('STATUS', f'SESSION_STOP:{self.current_session}'))
        self.status_queue.put(('INFO', f'Samples written: {self.samples_written}'))
        self.status_queue.put(('INFO', f'Inferences performed: {self.inference_count}'))
        
        self.current_session = None
        self.samples_written = 0
        self.inference_count = 0
        self._last_label          = None
        self._last_heartbeat      = 0.0
        self.recording_start_time = None  # stops heartbeat ticking after session ends
    
    def run_inference(self, processed_signal: np.ndarray) -> Dict:
        """
        Run TFLite inference on processed signal.
        
        Args:
            processed_signal: Filtered and normalized signal (1024 samples)
        
        Returns:
            dict with 'class', 'label', 'confidence'
        """
        # If model not loaded, return default NSR
        if self.interpreter is None:
            return {
                'class': 0,
                'label': 'NSR',
                'confidence': 0.00,
                'manual': False
            }
        
        try:
            if not self.input_details:
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.input_detail_map = {d['index']: d for d in self.input_details}
                self._resolve_input_tensors()

            # Primary CNN branch input (signal tensor)
            signal_detail = self.input_detail_map.get(self.signal_input_index, self.input_details[0])
            signal_data = np.asarray(processed_signal, dtype=np.float32).reshape(1, 1024, 1)
            signal_shape = tuple(int(v) for v in signal_detail.get('shape', signal_data.shape))
            if len(signal_shape) == 2:
                signal_data = signal_data.reshape(1, 1024)
            elif len(signal_shape) == 3:
                signal_data = signal_data.reshape(1, 1024, 1)
            signal_tensor = self._quantize_if_needed(signal_data, signal_detail)
            self.interpreter.set_tensor(signal_detail['index'], signal_tensor)

            # Secondary MLP branch input (RR feature tensor)
            if self.feature_input_index is not None:
                feature_detail = self.input_detail_map[self.feature_input_index]
                rr_features = self._extract_rr_features(processed_signal).reshape(1, RR_FEATURE_COUNT)

                feature_shape = tuple(int(v) for v in feature_detail.get('shape', rr_features.shape))
                if len(feature_shape) == 2 and feature_shape[-1] > 0:
                    expected = feature_shape[-1]
                    if rr_features.shape[1] > expected:
                        rr_features = rr_features[:, :expected]
                    elif rr_features.shape[1] < expected:
                        rr_features = np.pad(rr_features, ((0, 0), (0, expected - rr_features.shape[1])))

                feature_tensor = self._quantize_if_needed(rr_features, feature_detail)
                self.interpreter.set_tensor(feature_detail['index'], feature_tensor)

            # Run inference
            self.interpreter.invoke()

            # Get output tensor (probabilities for 4 classes)
            if not self.output_details:
                self.output_details = self.interpreter.get_output_details()

            out_detail = self.output_details[0]
            output_data = self.interpreter.get_tensor(out_detail['index'])
            output_probs = self._dequantize_if_needed(output_data, out_detail)
            probs_row = output_probs[0] if output_probs.ndim > 1 else output_probs

            # Extract predicted class and confidence
            predicted_class = int(np.argmax(probs_row))
            confidence = float(np.max(probs_row))
            
            return {
                'class': predicted_class,
                'label': RHYTHM_CLASSES.get(predicted_class, 'NSR'),
                'confidence': confidence,
                'manual': False
            }
        
        except Exception as e:
            self.status_queue.put(('ERROR', f'Inference failed: {e}'))
            # Return default NSR on error
            return {
                'class': 0,
                'label': 'NSR',
                'confidence': 0.00,
                'manual': False
            }
    
    def write_signal_samples(self, segment: Dict):
        """
        Write raw and filtered signal samples to CSV files.
        
        Args:
            segment: Segment containing timestamp_ms, raw ADC values, and filtered signal
        """
        if not self.signal_writer:
            return
        
        timestamps = segment['timestamp_ms']
        raw_values = segment['raw_signal']
        filtered_values = segment['filtered_signal']
        
        # Write raw signal
        for timestamp_ms, adc_value in zip(timestamps, raw_values):
            self.signal_writer.writerow([timestamp_ms, adc_value])
        
        # Write filtered signal with original timestamps (filtfilt is zero-phase).
        if self.filtered_writer:
            for timestamp_ms, filtered_value in zip(timestamps, filtered_values):
                self.filtered_writer.writerow([timestamp_ms, filtered_value])
        
        self.samples_written += len(timestamps)
        
        # Flush every 10 segments for safety
        if self.inference_count % 10 == 0:
            self.signal_file.flush()
            if self.filtered_file:
                self.filtered_file.flush()
    
    def write_rhythm_annotation(self, segment: Dict, prediction: Dict):
        """
        Write rhythm annotation to file.
        Includes manual override indicator in annotation.
        
        Args:
            segment: Segment metadata
            prediction: Inference result (with optional 'manual' flag)
        """
        if not self.annotation_writer:
            return
        
        label = prediction['label']
        
        self.annotation_writer.writerow([
            segment['segment_id'],
            segment['start_time'],
            segment['end_time'],
            prediction['class'],
            label,
            f"{prediction['confidence']:.4f}"
        ])
        
        # Calculate current duration
        if self.recording_start_time:
            current_duration = int(time.time() - self.recording_start_time)
            # Send rhythm classification with duration to OLED display
            self.status_queue.put(('STATUS', f"RHYTHM:{prediction['label']}:{current_duration}"))
            # Cache for 1-second heartbeat updates
            self._last_label     = prediction['label']
            self._last_heartbeat = time.time()
        else:
            # Fallback without duration
            self.status_queue.put(('STATUS', f"RHYTHM:{prediction['label']}"))
        
        # Trigger buzzer for abnormal rhythms
        if prediction['class'] != 0:  # Non-NSR (AFIB, PVC, LBBB)
            self._buzzer_alert(prediction['label'])
        
        # Flush every 10 segments
        if self.inference_count % 10 == 0:
            self.annotation_file.flush()
    
    def _buzzer_alert(self, rhythm_label: str):
        """
        Trigger buzzer pattern based on abnormal rhythm type.
        
        Alarm patterns (Morse code):
        - AFib: Short-Short-Short (S in Morse: ...)
        - PVC: Short-Long (A in Morse: .-)
        - LBBB: Very Long (T in Morse: -)
        
        Args:
            rhythm_label: Detected rhythm (AFIB, PVC, or LBBB)
        """
        if not self.buzzer:
            return
        
        try:
            if rhythm_label == 'AFIB':
                # AFIB: ... (3 short beeps)
                for i in range(3):
                    self.buzzer.on()
                    time.sleep(SHORT_BEEP)
                    self.buzzer.off()
                    if i < 2:  # Gap between beeps, not after last one
                        time.sleep(ELEMENT_GAP)
                        
            elif rhythm_label == 'PVC':
                # PVC: .- (1 short + 1 long)
                self.buzzer.on()
                time.sleep(SHORT_BEEP)
                self.buzzer.off()
                time.sleep(ELEMENT_GAP)
                self.buzzer.on()
                time.sleep(LONG_BEEP)
                self.buzzer.off()
                
            elif rhythm_label == 'LBBB':
                # LBBB: - (1 very long beep)
                self.buzzer.on()
                time.sleep(VERY_LONG_BEEP)
                self.buzzer.off()
                
        except Exception as e:
            self.status_queue.put(('WARN', f'Buzzer error: {e}'))
    
    def process_segment(self, segment: Dict):
        """
        Process one segment: inference + file writing.
        
        Args:
            segment: Processed segment from filtering
        """
        # Run inference on processed signal
        prediction = self.run_inference(segment['processed_signal'])
        
        # Write raw and filtered signals to CSV
        self.write_signal_samples(segment)
        
        # Write rhythm annotation
        self.write_rhythm_annotation(segment, prediction)
        
        self.inference_count += 1
    
    def inference_loop(self):
        """
        Main inference loop:
        - Receive processed segments
        - Run TFLite inference
        - Write signal and annotation files
        """
        self.status_queue.put(('INFO', 'Inference process started'))
        
        session_active = False
        current_session_id = 0  # Track which recording session we're processing
        
        while self.running:
            # Check for control messages (non-blocking)
            try:
                msg = self.control_queue.get_nowait()
                if msg.startswith('NEW_SESSION:'):
                    # New recording session starting
                    new_session_id = int(msg.split(':')[1])
                    current_session_id = new_session_id
                    self.status_queue.put(('DEBUG', f'New session ID: {current_session_id}'))
                    # Anchor the timer to acquisition start — first segment arrives
                    # ~4.76 s later, so the OLED timer and BUFFERING screen both
                    # reflect real elapsed recording time from the very first sample.
                    self.recording_start_time = time.time()
                    self._last_label     = None
                    self._last_heartbeat = 0.0
                elif msg == 'FLUSH_QUEUES':
                    # Clear inference queue of stale data from previous session
                    flushed = 0
                    while True:
                        try:
                            self.inference_queue.get_nowait()
                            flushed += 1
                        except mp.queues.Empty:
                            break
                    if flushed > 0:
                        self.status_queue.put(('DEBUG', f'Flushed {flushed} stale segments from inference queue'))
                elif msg == 'STOP_SESSION':
                    if session_active:
                        # Drain remaining segments from queue before closing
                        drained = 0
                        while True:
                            try:
                                segment = self.inference_queue.get_nowait()
                                self.process_segment(segment)
                                drained += 1
                            except mp.queues.Empty:
                                break
                        
                        if drained > 0:
                            self.status_queue.put(('DEBUG', f'Drained {drained} segments before session close'))
                        
                        # Close current session
                        self.stop_session()
                        session_active = False
                    else:
                        # Recording stopped before any segment arrived (e.g. button
                        # pressed during BUFFERING).  Reset timer state so the OLED
                        # heartbeat stops and we return to idle cleanly.
                        self.recording_start_time = None
                        self._last_label = None
                        self._last_heartbeat = 0.0
                        self.status_queue.put(('STATUS', 'SAVED:cancelled:0'))
            except mp.queues.Empty:
                pass
            
            try:
                # Get processed segment (with timeout)
                segment = self.inference_queue.get(timeout=0.1)
                
                # Check if segment is from current session
                segment_session_id = segment.get('session_id', 0)
                if segment_session_id != current_session_id:
                    # Stale segment from previous recording - discard
                    continue
                
                # Start new session if needed
                if not session_active:
                    timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
                    self.start_session(f"ecg_{timestamp}")
                    session_active = True
                
                # Process segment
                self.process_segment(segment)
                
                # Status update every 10 inferences
                if self.inference_count % 10 == 0:
                    self.status_queue.put(('DEBUG', f'Inferences: {self.inference_count}'))
            
            except mp.queues.Empty:
                # No segment arrived — tick the OLED timer every second.
                # During BUFFERING (_last_label is None) send BUFFERING:{n} so
                # the OLED counts up. After first result, send RHYTHM:{label}:{n}.
                if (self.recording_start_time is not None
                        and (time.time() - self._last_heartbeat) >= 1.0):
                    current_duration = int(time.time() - self.recording_start_time)
                    if self._last_label is not None:
                        self.status_queue.put(('STATUS', f'RHYTHM:{self._last_label}:{current_duration}'))
                    else:
                        self.status_queue.put(('STATUS', f'BUFFERING:{current_duration}'))
                    self._last_heartbeat = time.time()
                continue
            except Exception as e:
                self.status_queue.put(('ERROR', f'Inference error: {e}'))
        
        # Close session on shutdown
        if session_active:
            self.stop_session()
    
    def run(self):
        """Process main entry point"""
        self.setup()
        self.running = True
        
        try:
            self.inference_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.signal_file:
            self.signal_file.close()
        if self.filtered_file:
            self.filtered_file.close()
        if self.annotation_file:
            self.annotation_file.close()
        
        # Cleanup buzzer
        if self.buzzer:
            try:
                self.buzzer.off()
                self.buzzer.close()
            except:
                pass
        
        self.status_queue.put(('INFO', 'Inference process terminated'))


def inference_worker(inference_queue: mp.Queue, status_queue: mp.Queue,
                    control_queue: mp.Queue, output_dir: Path):
    """
    Worker function for inference process.
    
    Args:
        inference_queue: Input queue from filtering
        status_queue: Output queue for status messages
        control_queue: Input queue for control commands
        output_dir: Directory for output files
    """
    process = InferenceProcess(inference_queue, status_queue, control_queue, output_dir)
    process.run()
