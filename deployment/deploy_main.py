#!/usr/bin/env python3
"""
ECG Acquisition Pipeline - Main Orchestrator
Multi-process architecture for real-time ECG acquisition and inference.

Process Distribution (4 cores):
  Core 0: Acquisition + GPIO
  Core 1: Segmentation
  Core 2: Filtering + Normalization
  Core 3: Inference + File I/O
  
Pipeline Flow:
  Acquisition → Segmentation → Filtering → Inference
"""

import multiprocessing as mp
import signal
import sys
import time
from pathlib import Path
from typing import Dict

from deploy_acquisition import acquisition_worker
from deploy_segmentation import segmentation_worker
from deploy_filtering import filtering_worker
from deploy_inference import inference_worker
from deploy_oled import oled_worker

OUTPUT_DIR = Path('deployment_data')
QUEUE_MAXSIZE = 100

class ECGPipeline:
    """Main orchestrator for ECG acquisition pipeline."""
    
    def __init__(self):
        self.packet_queue = mp.Queue(maxsize=QUEUE_MAXSIZE)
        self.segment_queue = mp.Queue(maxsize=QUEUE_MAXSIZE)
        self.inference_queue = mp.Queue(maxsize=QUEUE_MAXSIZE)
        self.status_queue = mp.Queue()
        self.oled_status_queue = mp.Queue()
        
        self.acq_control = mp.Queue()
        self.seg_control = mp.Queue()
        self.filt_control = mp.Queue()
        self.inf_control = mp.Queue()
        self.oled_control = mp.Queue()
        
        self.processes: Dict[str, mp.Process] = {}
        self.running = False
    
    def setup_processes(self):
        """Initialize all worker processes."""
        self.processes['oled'] = mp.Process(
            target=oled_worker,
            args=(self.oled_status_queue, self.oled_control),
            name='OLED'
        )
        
        self.processes['acquisition'] = mp.Process(
            target=acquisition_worker,
            args=(self.packet_queue, self.status_queue, self.acq_control, self.inf_control),
            name='Acquisition'
        )
        
        self.processes['segmentation'] = mp.Process(
            target=segmentation_worker,
            args=(self.packet_queue, self.segment_queue, self.status_queue, self.seg_control),
            name='Segmentation'
        )
        
        self.processes['filtering'] = mp.Process(
            target=filtering_worker,
            args=(self.segment_queue, self.inference_queue, self.status_queue, 
                  self.filt_control, str(OUTPUT_DIR)),
            name='Filtering'
        )
        
        self.processes['inference'] = mp.Process(
            target=inference_worker,
            args=(self.inference_queue, self.status_queue, self.inf_control, OUTPUT_DIR),
            name='Inference'
        )
    
    def start_pipeline(self):
        """Start all processes."""
        print("[PIPELINE] Starting ECG acquisition pipeline...")
        print(f"[PIPELINE] Output directory: {OUTPUT_DIR.absolute()}")
        print(f"[PIPELINE] Process architecture:")
        print(f"  └─ OLED Display")
        print(f"  └─ Core 0: Acquisition + GPIO")
        print(f"  └─ Core 1: Segmentation")
        print(f"  └─ Core 2: Filtering + Normalization")
        print(f"  └─ Core 3: Inference + File I/O")
        print()
        
        self.running = True
        
        for name, process in self.processes.items():
            process.start()
            print(f"[PIPELINE] Started {name} process (PID: {process.pid})")
        
        print()
        print("[PIPELINE] All processes started")
        print("[PIPELINE] Press button to start/stop recording")
        print("[PIPELINE] Press Ctrl+C to exit")
        print()
    
    def status_monitor_loop(self):
        """Monitor status messages from all processes."""
        while self.running:
            try:
                msg_type, msg_content = self.status_queue.get(timeout=0.1)
                timestamp = time.strftime('%H:%M:%S')
                
                self.oled_status_queue.put((msg_type, msg_content))
                
                if msg_type == 'STATUS':
                    print(f"[{timestamp}] ● {msg_content}")
                elif msg_type == 'INFO':
                    print(f"[{timestamp}]   {msg_content}")
                elif msg_type == 'WARN':
                    print(f"[{timestamp}] ⚠ {msg_content}")
                elif msg_type == 'ERROR':
                    print(f"[{timestamp}] ✗ {msg_content}")
                elif msg_type == 'DEBUG':
                    print(f"[{timestamp}] [DEBUG] {msg_content}")
            
            except mp.queues.Empty:
                continue
            except KeyboardInterrupt:
                break
    
    def shutdown_pipeline(self):
        """Gracefully shutdown all processes."""
        print("\n[PIPELINE] Shutting down...")
        
        self.running = False
        
        timeout = 5.0
        for name, process in self.processes.items():
            if process.is_alive():
                print(f"[PIPELINE] Waiting for {name} to terminate...")
                process.join(timeout=timeout)
                
                if process.is_alive():
                    print(f"[PIPELINE] Force terminating {name}...")
                    process.terminate()
                    process.join(timeout=1.0)
        
        for queue in [self.packet_queue, self.segment_queue, self.inference_queue, 
                     self.status_queue, self.oled_status_queue, self.acq_control, 
                     self.seg_control, self.filt_control, self.inf_control, self.oled_control]:
            queue.close()
            queue.join_thread()
        
        print("[PIPELINE] Shutdown complete")
    
    def run(self):
        """Main pipeline execution."""
        def signal_handler(sig, frame):
            self.shutdown_pipeline()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.setup_processes()
        self.start_pipeline()
        
        try:
            self.status_monitor_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown_pipeline()


def main():
    """Entry point."""
    print("=" * 60)
    print("ECG ACQUISITION PIPELINE - DEPLOYMENT VERSION")
    print("=" * 60)
    print()
    
    pipeline = ECGPipeline()
    pipeline.run()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
