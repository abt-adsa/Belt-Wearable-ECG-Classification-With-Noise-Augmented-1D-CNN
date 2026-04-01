#!/usr/bin/env python3
"""
Main Deployment Script
"""

import multiprocessing as mp
import signal
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict

# Import worker processes
from deploy_acquisition import acquisition_worker
from deploy_segmentation import segmentation_worker
from deploy_filtering import filtering_worker
from deploy_inference import inference_worker
from deploy_oled import oled_worker

# Configuration
OUTPUT_DIR = Path.home() / 'thesis' / 'data'
QUEUE_MAXSIZE = 100  # Prevent unbounded memory growth

class ECGPipeline:
    """
    Main orchestrator for ECG acquisition pipeline.
    Manages processes and inter-process communication.
    """
    
    def __init__(self):
        # Reset I2C pins BEFORE creating any processes
        self._reset_i2c_pins()
        
        # Inter-process queues
        self.packet_queue = mp.Queue(maxsize=QUEUE_MAXSIZE)
        self.segment_queue = mp.Queue(maxsize=QUEUE_MAXSIZE)
        self.inference_queue = mp.Queue(maxsize=QUEUE_MAXSIZE)
        self.status_queue = mp.Queue()  # No size limit for status messages
        self.oled_status_queue = mp.Queue()  # Dedicated queue for OLED display
        
        # Control queues (for future use)
        self.acq_control = mp.Queue()
        self.seg_control = mp.Queue()
        self.filt_control = mp.Queue()
        self.inf_control = mp.Queue()
        self.oled_control = mp.Queue()
        
        # Process handles
        self.processes: Dict[str, mp.Process] = {}
        
        # Pipeline state
        self.running = False
    
    def _reset_i2c_pins(self):
        """
        Reset I2C pins to ALT0 mode before starting any processes.
        Prevents GPIO library from changing I2C pins to GPIO mode.
        """
        try:
            # GPIO2 (SDA) and GPIO3 (SCL) - WiringPi pins 8 and 9
            subprocess.run(['gpio', 'mode', '8', 'ALT0'], check=False, capture_output=True, timeout=1)
            subprocess.run(['gpio', 'mode', '9', 'ALT0'], check=False, capture_output=True, timeout=1)
            time.sleep(0.1)
        except:
            pass  # Best effort - continue even if this fails
    
    def setup_processes(self):
        """Initialize all worker processes"""
        
        # Process 1: OLED Display (independent, visual feedback)
        self.processes['oled'] = mp.Process(
            target=oled_worker,
            args=(self.oled_status_queue, self.oled_control),
            name='OLED'
        )
        
        # Process 2: Acquisition + GPIO (Core 0)
        self.processes['acquisition'] = mp.Process(
            target=acquisition_worker,
            args=(self.packet_queue, self.status_queue, self.acq_control, self.inf_control),
            name='Acquisition'
        )
        
        # Process 4: Segmentation (Core 1)
        self.processes['segmentation'] = mp.Process(
            target=segmentation_worker,
            args=(self.packet_queue, self.segment_queue, self.status_queue, self.seg_control),
            name='Segmentation'
        )
        
        # Process 5: Filtering (Core 2)
        self.processes['filtering'] = mp.Process(
            target=filtering_worker,
            args=(self.segment_queue, self.inference_queue, self.status_queue, 
                  self.filt_control, str(OUTPUT_DIR)),
            name='Filtering'
        )
        
        # Process 6: Inference + I/O (Core 3)
        self.processes['inference'] = mp.Process(
            target=inference_worker,
            args=(self.inference_queue, self.status_queue, self.inf_control, 
                  OUTPUT_DIR),
            name='Inference'
        )
    
    def start_pipeline(self):
        """Start all processes"""
        self.running = True
        
        # Start OLED first BEFORE GPIO init (to avoid I2C pin conflicts)
        # Then start other processes
        start_order = ['oled', 'segmentation', 'filtering', 'inference', 'acquisition']
        
        for name in start_order:
            if name in self.processes:
                process = self.processes[name]
                process.start()
                if name == 'oled':
                    time.sleep(0.5)  # Give OLED time to claim I2C before GPIO init
    
    def status_monitor_loop(self):
        """
        Monitor status messages from all processes.
        Runs in main thread to display pipeline status.
        Also forwards relevant messages to OLED display and Flask server.
        """
        _init_seen = set()
        _INIT_GATES = {'Inference process initialized', 'Acquisition process initialized'}
        _init_ready_sent = False

        while self.running:
            try:
                # Check for status messages (non-blocking)
                msg_type, msg_content = self.status_queue.get(timeout=0.1)

                timestamp = time.strftime('%H:%M:%S')

                # Track process-init milestones and emit INIT_READY once all are up
                if not _init_ready_sent and msg_type == 'INFO' and msg_content in _INIT_GATES:
                    _init_seen.add(msg_content)
                    if _init_seen >= _INIT_GATES:
                        self.oled_status_queue.put(('STATUS', 'INIT_READY'))
                        _init_ready_sent = True

                # Forward every message to OLED display
                self.oled_status_queue.put((msg_type, msg_content))

                # Format output based on message type
                if msg_type == 'STATUS':
                    print(f"[{timestamp}] ● {msg_content}")
                elif msg_type == 'INFO':
                    print(f"[{timestamp}]   {msg_content}")
                elif msg_type == 'WARN':
                    print(f"[{timestamp}] ⚠ {msg_content}")
                elif msg_type == 'ERROR':
                    print(f"[{timestamp}] ✗ {msg_content}")
                elif msg_type == 'DEBUG':
                    # Print debug messages to see packet flow
                    print(f"[{timestamp}] [DEBUG] {msg_content}")
            
            except mp.queues.Empty:
                continue
            except KeyboardInterrupt:
                break
    
    def shutdown_pipeline(self):
        """Gracefully shutdown all processes"""
        print("\n[PIPELINE] Shutting down...")
        
        self.running = False
        
        # Wait for processes to finish gracefully
        timeout = 5.0
        for name, process in self.processes.items():
            if process.is_alive():
                print(f"[PIPELINE] Waiting for {name} to terminate...")
                process.join(timeout=timeout)
                
                if process.is_alive():
                    print(f"[PIPELINE] Force terminating {name}...")
                    process.terminate()
                    process.join(timeout=1.0)
        
        # Close queues
        for queue in [self.packet_queue, self.segment_queue, self.inference_queue, 
                     self.status_queue, self.oled_status_queue,
                     self.acq_control, self.seg_control, self.filt_control, 
                     self.inf_control, self.oled_control]:
            queue.close()
            queue.join_thread()
        
        print("[PIPELINE] Shutdown complete")
    
    def run(self):
        """Main pipeline execution"""
        # Setup signal handler for graceful shutdown
        def signal_handler(sig, frame):
            self.shutdown_pipeline()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Setup and start processes
        self.setup_processes()
        self.start_pipeline()
        
        # Monitor status messages
        try:
            self.status_monitor_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown_pipeline()


def main():
    """Entry point"""
    print("=" * 60)
    print("ECG ACQUISITION PIPELINE - DEPLOYMENT VERSION")
    print("=" * 60)
    print()
    
    pipeline = ECGPipeline()
    pipeline.run()


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method('spawn', force=True)
    main()
