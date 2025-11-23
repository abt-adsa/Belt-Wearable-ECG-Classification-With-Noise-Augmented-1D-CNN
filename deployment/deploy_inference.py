#!/usr/bin/env python3
"""
Inference Process - TFLite inference and file I/O management.
- TFLite inference (4 rhythm classes)
- Write signal CSV (timestamp_ms, adc_value)
- Write rhythm annotation file (synchronized with signal)
"""

import multiprocessing as mp
import csv
import time
from pathlib import Path
from datetime import datetime
import numpy as np
from gpiozero import DigitalOutputDevice

RHYTHM_CLASSES = {
    0: 'NSR',
    1: 'AFIB',
    2: 'PVC',
    3: 'LBBB'
}

BUZZER_PIN = 6

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
        
        self.signal_file = None
        self.signal_writer = None
        self.filtered_file = None
        self.filtered_writer = None
        self.annotation_file = None
        self.annotation_writer = None
        self.current_session = None
        
        self.buzzer = None
        
        self.interpreter = None
        
        self.inference_count = 0
        self.samples_written = 0
        
        self.recording_start_time = None
        self.recording_duration = 0
    
    def setup(self):
        """Initialize resources."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.interpreter = self.load_tflite_model()
        
        try:
            self.buzzer = DigitalOutputDevice(BUZZER_PIN)
            self.status_queue.put(('INFO', f'Buzzer initialized on GPIO {BUZZER_PIN}'))
        except Exception as e:
            self.status_queue.put(('WARN', f'Buzzer init failed: {e}'))
            self.buzzer = None
        
        self.status_queue.put(('INFO', 'Inference process initialized'))
    
    def load_tflite_model(self):
        """Load TFLite model for inference."""
        from tflite_runtime.interpreter import Interpreter
        
        model_path = Path(__file__).parent / 'model_zscore.tflite'
        
        if not model_path.exists():
            self.status_queue.put(('ERROR', f'Model not found: {model_path}'))
            return None
        
        try:
            interpreter = Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            self.status_queue.put(('INFO', f'Loaded TFLite model: {model_path}'))
            return interpreter
        except Exception as e:
            self.status_queue.put(('ERROR', f'Failed to load model: {e}'))
            return None
    
    def start_session(self, session_name: str):
        """Start a new recording session."""
        self.current_session = session_name
        self.recording_start_time = time.time()
        self.recording_duration = 0
        
        signal_path = self.output_dir / f"{session_name}_signal.csv"
        self.signal_file = open(signal_path, 'w', newline='')
        self.signal_writer = csv.writer(self.signal_file)
        self.signal_writer.writerow(['timestamp_ms', 'adc_value'])
        
        filtered_path = self.output_dir / f"{session_name}_filtered.csv"
        self.filtered_file = open(filtered_path, 'w', newline='')
        self.filtered_writer = csv.writer(self.filtered_file)
        self.filtered_writer.writerow(['timestamp_ms', 'adc_value'])
        
        annotation_path = self.output_dir / f"{session_name}_rhythm.csv"
        self.annotation_file = open(annotation_path, 'w', newline='')
        self.annotation_writer = csv.writer(self.annotation_file)
        self.annotation_writer.writerow(['segment_id', 'start_time_ms', 'end_time_ms', 
                                        'rhythm_class', 'rhythm_label', 'confidence'])
        
        self.status_queue.put(('STATUS', f'SESSION_START:{session_name}'))
        self.status_queue.put(('INFO', f'Signal file: {signal_path}'))
        self.status_queue.put(('INFO', f'Rhythm file: {annotation_path}'))
    
    def stop_session(self):
        """Stop current recording session and close files."""
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
        
        self.status_queue.put(('STATUS', f'SAVED:{self.current_session}:{self.recording_duration}'))
        
        self.status_queue.put(('STATUS', f'SESSION_STOP:{self.current_session}'))
        self.status_queue.put(('INFO', f'Samples written: {self.samples_written}'))
        self.status_queue.put(('INFO', f'Inferences performed: {self.inference_count}'))
        
        self.current_session = None
        self.samples_written = 0
        self.inference_count = 0
    
    def run_inference(self, processed_signal: np.ndarray) -> dict:
        """Run TFLite inference on processed signal."""
        if self.interpreter is None:
            return {
                'class': 0,
                'label': 'NSR',
                'confidence': 0.00
            }
        
        try:
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            input_data = processed_signal.reshape(1, 1024, 1).astype(np.float32)
            
            self.interpreter.set_tensor(input_details[0]['index'], input_data)
            
            self.interpreter.invoke()
            
            output_data = self.interpreter.get_tensor(output_details[0]['index'])
            
            predicted_class = int(np.argmax(output_data[0]))
            confidence = float(np.max(output_data[0]))
            
            return {
                'class': predicted_class,
                'label': RHYTHM_CLASSES[predicted_class],
                'confidence': confidence
            }
        
        except Exception as e:
            self.status_queue.put(('ERROR', f'Inference failed: {e}'))
            return {
                'class': 0,
                'label': 'NSR',
                'confidence': 0.00
            }
    
    def write_signal_samples(self, segment: dict):
        """Write raw and filtered signal samples to CSV files."""
        if not self.signal_writer:
            return
        
        timestamps = segment['timestamp_ms']
        raw_values = segment['raw_signal']
        filtered_values = segment['filtered_signal']
        
        for timestamp_ms, adc_value in zip(timestamps, raw_values):
            self.signal_writer.writerow([timestamp_ms, adc_value])
        
        if self.filtered_writer:
            for timestamp_ms, filtered_value in zip(timestamps, filtered_values):
                self.filtered_writer.writerow([timestamp_ms, filtered_value])
        
        self.samples_written += len(timestamps)
        
        if self.inference_count % 10 == 0:
            self.signal_file.flush()
            if self.filtered_file:
                self.filtered_file.flush()
    
    def _buzzer_alert(self, rhythm_class: int):
        """Generate buzzer patterns for rhythm abnormalities."""
        if rhythm_class == 1:
            for _ in range(3):
                self.buzzer.on()
                time.sleep(0.15)
                self.buzzer.off()
                time.sleep(0.1)
        elif rhythm_class == 2:
            for _ in range(2):
                self.buzzer.on()
                time.sleep(0.15)
                self.buzzer.off()
                time.sleep(0.1)
        elif rhythm_class == 3:
            self.buzzer.on()
            time.sleep(0.3)
            self.buzzer.off()
    
    def write_rhythm_annotation(self, segment: dict, prediction: dict):
        """Write rhythm annotation to file."""
        if not self.annotation_writer:
            return
        
        self.annotation_writer.writerow([
            segment['segment_id'],
            segment['start_time'],
            segment['end_time'],
            prediction['class'],
            prediction['label'],
            f"{prediction['confidence']:.4f}"
        ])
        
        if prediction['class'] != 0 and self.buzzer:
            try:
                self._buzzer_alert(prediction['class'])
            except Exception as e:
                self.status_queue.put(('DEBUG', f'Buzzer alert failed: {e}'))
        
        if self.recording_start_time:
            current_duration = int(time.time() - self.recording_start_time)
            self.status_queue.put(('STATUS', f"RHYTHM:{prediction['label']}:{current_duration}"))
        else:
            self.status_queue.put(('STATUS', f"RHYTHM:{prediction['label']}"))
        
        if self.inference_count % 10 == 0:
            self.annotation_file.flush()
    
    def process_segment(self, segment: dict):
        """Process one segment: inference + file writing."""
        prediction = self.run_inference(segment['processed_signal'])
        
        self.write_signal_samples(segment)
        
        self.write_rhythm_annotation(segment, prediction)
        
        self.inference_count += 1
    
    def inference_loop(self):
        """Main inference loop."""
        self.status_queue.put(('INFO', 'Inference process started'))
        
        session_active = False
        
        while self.running:
            try:
                msg = self.control_queue.get_nowait()
                if msg == 'STOP_SESSION' and session_active:
                    self.stop_session()
                    session_active = False
            except mp.queues.Empty:
                pass
            
            try:
                segment = self.inference_queue.get(timeout=0.1)
                
                if not session_active:
                    timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
                    self.start_session(f"ecg_{timestamp}")
                    session_active = True
                
                self.process_segment(segment)
                
                if self.inference_count % 10 == 0:
                    self.status_queue.put(('DEBUG', f'Inferences: {self.inference_count}'))
            
            except mp.queues.Empty:
                continue
            except Exception as e:
                self.status_queue.put(('ERROR', f'Inference error: {e}'))
        
        if session_active:
            self.stop_session()
    
    def run(self):
        """Process main entry point."""
        self.setup()
        self.running = True
        
        try:
            self.inference_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        if self.signal_file:
            self.signal_file.close()
        if self.filtered_file:
            self.filtered_file.close()
        if self.annotation_file:
            self.annotation_file.close()
        if self.buzzer:
            self.buzzer.close()
        
        self.status_queue.put(('INFO', 'Inference process terminated'))


def inference_worker(inference_queue: mp.Queue, status_queue: mp.Queue,
                    control_queue: mp.Queue, output_dir: Path):
    """Worker function for inference process."""
    process = InferenceProcess(inference_queue, status_queue, control_queue, output_dir)
    process.run()
