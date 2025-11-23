#!/usr/bin/env python3
"""
Segmentation Process - Buffer packets into 1024-sample segments for TFLite.
- Receive 256-sample packets from acquisition
- Buffer into 1024-sample segments
- Add overlap for edge-artifact-free filtering
"""

import multiprocessing as mp
import struct
from typing import List, Tuple
import numpy as np

BUFFER_SIZE = 256
SEGMENT_SIZE = 1024
OVERLAP_SIZE = 256
EXTENDED_SEGMENT_SIZE = SEGMENT_SIZE + (2 * OVERLAP_SIZE)
STEP_SIZE = SEGMENT_SIZE
PACKETS_PER_SEGMENT = SEGMENT_SIZE // BUFFER_SIZE

class SegmentationProcess:
    """Segments packets into 1024-sample windows for TFLite inference."""
    
    def __init__(self, 
                 packet_queue: mp.Queue, 
                 segment_queue: mp.Queue,
                 status_queue: mp.Queue):
        self.packet_queue = packet_queue
        self.segment_queue = segment_queue
        self.status_queue = status_queue
        
        self.segment_buffer: List[Tuple[int, int]] = []
        self.overlap_buffer: List[Tuple[int, int]] = []
        self.segment_count = 0
        self.running = False
    
    def _unpack_packet(self, packet_data: bytes) -> List[Tuple[int, int]]:
        """Unpack binary packet into list of (timestamp_ms, adc_value) tuples."""
        samples = []
        for i in range(0, len(packet_data), 6):
            timestamp_ms, adc_value = struct.unpack('<IH', packet_data[i:i+6])
            samples.append((timestamp_ms, adc_value))
        
        return samples
    
    def _create_segment(self) -> dict:
        """Create a segment dictionary with overlap for edge-artifact-free filtering."""
        extended_samples = self.overlap_buffer + self.segment_buffer[:SEGMENT_SIZE + OVERLAP_SIZE]
        
        extended_timestamps = np.array([s[0] for s in extended_samples], dtype=np.uint32)
        extended_adc = np.array([s[1] for s in extended_samples], dtype=np.uint16)
        
        core_start = len(self.overlap_buffer)
        core_end = core_start + SEGMENT_SIZE
        core_timestamps = extended_timestamps[core_start:core_end]
        
        segment = {
            'segment_id': self.segment_count,
            'extended_timestamp_ms': extended_timestamps,
            'extended_adc_value': extended_adc,
            'timestamp_ms': core_timestamps,
            'core_start_idx': core_start,
            'core_end_idx': core_end,
            'start_time': core_timestamps[0],
            'end_time': core_timestamps[-1],
            'sample_count': SEGMENT_SIZE
        }
        
        self.segment_count += 1
        
        self.overlap_buffer = self.segment_buffer[SEGMENT_SIZE:SEGMENT_SIZE + OVERLAP_SIZE]
        self.segment_buffer = self.segment_buffer[SEGMENT_SIZE:]
        
        return segment
    
    def segmentation_loop(self):
        """Main segmentation loop."""
        self.status_queue.put(('INFO', 'Segmentation process started'))
        
        while self.running:
            try:
                packet_data, arrival_time = self.packet_queue.get(timeout=0.1)
                
                samples = self._unpack_packet(packet_data)
                self.segment_buffer.extend(samples)
                
                if len(self.segment_buffer) >= SEGMENT_SIZE + OVERLAP_SIZE:
                    segment = self._create_segment()
                    self.segment_queue.put(segment)
                    
                    if self.segment_count % 10 == 0:
                        self.status_queue.put(('DEBUG', f'Segments created: {self.segment_count}'))
            
            except mp.queues.Empty:
                continue
            except Exception as e:
                self.status_queue.put(('ERROR', f'Segmentation error: {e}'))
    
    def run(self):
        """Process main entry point."""
        self.running = True
        
        try:
            self.segmentation_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup and flush remaining samples."""
        if len(self.segment_buffer) > 0:
            self.status_queue.put(('INFO', f'Flushing {len(self.segment_buffer)} remaining samples'))
        
        self.status_queue.put(('INFO', f'Segmentation process terminated. Total segments: {self.segment_count}'))


def segmentation_worker(packet_queue: mp.Queue, segment_queue: mp.Queue, 
                       status_queue: mp.Queue, control_queue: mp.Queue):
    """Worker function for segmentation process."""
    process = SegmentationProcess(packet_queue, segment_queue, status_queue)
    process.run()
