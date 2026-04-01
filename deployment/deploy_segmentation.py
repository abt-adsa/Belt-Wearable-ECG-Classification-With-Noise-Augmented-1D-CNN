#!/usr/bin/env python3
"""
Process 2: Segmentation + Buffering (Core 1)
=============================================
Intermediate process:
- Receive 256-sample packets from acquisition
- Buffer into 1024-sample segments for TFLite
- Forward segments to filtering/normalization

4 packets = 1024 samples = 1 inference window
"""

import multiprocessing as mp
import struct
from typing import List, Tuple
import numpy as np

BUFFER_SIZE = 256  # Samples per packet
SEGMENT_SIZE = 1024  # TFLite input size (core segment)
OVERLAP_SIZE = 689   # 1.91 s context each side — required for 1.0 Hz ord-4 Butterworth HPF to settle (<1%)
EXTENDED_SEGMENT_SIZE = SEGMENT_SIZE + (2 * OVERLAP_SIZE)  # 2402 total
STEP_SIZE = SEGMENT_SIZE  # Non-overlapping step between segments
PACKETS_PER_SEGMENT = SEGMENT_SIZE // BUFFER_SIZE  # 4 packets

class SegmentationProcess:
    """
    Segments packets into 1024-sample windows for TFLite inference.
    """
    
    def __init__(self, 
                 packet_queue: mp.Queue, 
                 segment_queue: mp.Queue,
                 status_queue: mp.Queue):
        self.packet_queue = packet_queue
        self.segment_queue = segment_queue
        self.status_queue = status_queue
        
        # Segmentation buffer
        self.segment_buffer: List[Tuple[int, int]] = []  # [(timestamp_ms, adc_value), ...]
        self.overlap_buffer: List[Tuple[int, int]] = []  # Previous segment tail for overlap
        self.segment_count = 0
        self.current_session_id = 0  # Track current recording session
        self.running = False
    
    def _unpack_packet(self, packet_data: bytes) -> List[Tuple[int, int]]:
        """
        Unpack binary packet into list of (timestamp_ms, adc_value) tuples.
        
        Args:
            packet_data: 1536 bytes (256 samples × 6 bytes)
        
        Returns:
            List of (timestamp_ms, adc_value) tuples
        """
        samples = []
        for i in range(0, len(packet_data), 6):
            timestamp_ms, adc_value = struct.unpack('<IH', packet_data[i:i+6])
            samples.append((timestamp_ms, adc_value))
        
        return samples
    
    def _create_segment(self) -> dict:
        """
        Create a segment dictionary with overlap for edge-artifact-free filtering.

        Extended segment structure:
        [1440 past overlap] [1024 core segment] [1440 future overlap]
        Total: 2402 samples.

        The segment is NOT dispatched until SEGMENT_SIZE + OVERLAP_SIZE samples
        have accumulated in the buffer, so the future overlap is real acquired
        data (not predicted).  This introduces ~1.91 s of deliberate latency after
        the last sample of the core, which is necessary for the 1.0 Hz ord-4
        Butterworth HPF to settle at both window boundaries (<1% residual).
        """
        # Combine overlap buffer + current segment + next overlap
        extended_samples = self.overlap_buffer + self.segment_buffer[:SEGMENT_SIZE + OVERLAP_SIZE]
        
        # Extract timestamps and ADC values for extended segment
        extended_timestamps = np.array([s[0] for s in extended_samples], dtype=np.uint32)
        extended_adc = np.array([s[1] for s in extended_samples], dtype=np.uint16)
        
        # Core segment timestamps (for final output)
        core_start = len(self.overlap_buffer)
        core_end = core_start + SEGMENT_SIZE
        core_timestamps = extended_timestamps[core_start:core_end]
        
        # Gap detection: check for timestamp discontinuities
        expected_interval_ms = 1000.0 / 360.0  # ~2.78ms for 360 Hz
        if len(core_timestamps) > 1:
            timestamp_diffs = np.diff(core_timestamps.astype(np.float64))
            gap_threshold = expected_interval_ms * 3  # Allow 3x tolerance
            gaps = np.where(timestamp_diffs > gap_threshold)[0]
            
            if len(gaps) > 0:
                total_gap_duration = np.sum(timestamp_diffs[gaps] - expected_interval_ms)
                self.status_queue.put((
                    'WARN', 
                    f'Segment {self.segment_count}: Detected {len(gaps)} gaps, '
                    f'total missing time: {total_gap_duration:.1f}ms'
                ))
        
        segment = {
            'segment_id': self.segment_count,
            'session_id': self.current_session_id,  # Mark with current session
            # Extended data for filtering (with overlap)
            'extended_timestamp_ms': extended_timestamps,
            'extended_adc_value': extended_adc,
            # Core segment metadata (for final output)
            'timestamp_ms': core_timestamps,
            'core_start_idx': core_start,
            'core_end_idx': core_end,
            'start_time': core_timestamps[0],
            'end_time': core_timestamps[-1],
            'sample_count': SEGMENT_SIZE
        }
        
        self.segment_count += 1
        
        # Save last 256 samples as overlap for next segment
        self.overlap_buffer = self.segment_buffer[SEGMENT_SIZE:SEGMENT_SIZE + OVERLAP_SIZE]
        
        # Remove processed samples, keep remainder
        self.segment_buffer = self.segment_buffer[SEGMENT_SIZE:]
        
        return segment
    
    def segmentation_loop(self):
        """
        Main segmentation loop:
        - Receive packets from acquisition
        - Buffer into 1024-sample segments
        - Forward to filtering process
        """
        self.status_queue.put(('INFO', 'Segmentation process started'))
        
        while self.running:
            try:
                # Get packet from acquisition (with timeout for clean shutdown)
                packet_data, arrival_time, session_id = self.packet_queue.get(timeout=0.1)
                
                # Update session ID if it changed (new recording started)
                if session_id != self.current_session_id:
                    self.current_session_id = session_id
                    # Reset buffers for new session
                    self.segment_buffer = []
                    self.overlap_buffer = []
                    self.segment_count = 0
                    self.status_queue.put(('DEBUG', f'Segmentation: New session {session_id}'))
                
                # Unpack samples
                samples = self._unpack_packet(packet_data)
                
                # Add to segment buffer
                self.segment_buffer.extend(samples)
                
                # Check if we have enough samples (1024 core + 256 next overlap)
                if len(self.segment_buffer) >= SEGMENT_SIZE + OVERLAP_SIZE:
                    # Create segment with overlap context
                    segment = self._create_segment()
                    
                    # Forward to filtering process
                    self.segment_queue.put(segment)
                    
                    # Status update every 10 segments
                    if self.segment_count % 10 == 0:
                        self.status_queue.put(('DEBUG', f'Segments created: {self.segment_count}'))
            
            except mp.queues.Empty:
                continue  # No packets available, continue waiting
            except Exception as e:
                self.status_queue.put(('ERROR', f'Segmentation error: {e}'))
    
    def run(self):
        """Process main entry point"""
        self.running = True
        
        try:
            self.segmentation_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup and flush remaining samples"""
        # If we have remaining samples in buffer, create final segment
        if len(self.segment_buffer) > 0:
            self.status_queue.put(('INFO', f'Flushing {len(self.segment_buffer)} remaining samples'))
            # For production: might want to pad to 1024 or handle partial segments
        
        self.status_queue.put(('INFO', f'Segmentation process terminated. Total segments: {self.segment_count}'))


def segmentation_worker(packet_queue: mp.Queue, segment_queue: mp.Queue, 
                       status_queue: mp.Queue, control_queue: mp.Queue):
    """
    Worker function for segmentation process.
    
    Args:
        packet_queue: Input queue from acquisition
        segment_queue: Output queue to filtering
        status_queue: Output queue for status messages
        control_queue: Input queue for control commands
    """
    process = SegmentationProcess(packet_queue, segment_queue, status_queue)
    process.run()
