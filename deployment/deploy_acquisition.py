#!/usr/bin/env python3
"""
Process 1: Acquisition + GPIO (Core 0)
=======================================
Real-time critical process:
- USB serial acquisition (MARKER-based binary packets)
- Button/LED control
- Forward packets to segmentation process

Runs on dedicated core for minimal latency.
"""

import serial
import time
import struct
import subprocess
import multiprocessing as mp
import threading
from gpiozero import Button
from typing import Optional

# Configuration
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
BUFFER_SIZE = 256
MARKER = b"MARKER"
PACKET_DATA_SIZE = BUFFER_SIZE * 6  # 1536 bytes
PACKET_SIZE = len(MARKER) + PACKET_DATA_SIZE  # 1542 bytes

# GPIO Pin Assignments
# Button Label 1: Recording on/off
BUTTON_PIN = 26
# Button Label 2: Data transfer to USB
TRANSFER_BUTTON_PIN = 19
# Button Label 3: Shutdown (immediate)
SHUTDOWN_BUTTON_PIN = 13
BUTTON_DEBOUNCE_TIME = 0.3  # 300ms debounce to prevent double-press
RECORDING_COUNTDOWN_SEC = 3  # 3-second countdown before recording starts

class AcquisitionProcess:
    """
    Acquisition + GPIO control process.
    Forwards raw packets to segmentation process.
    Uses OLED for status feedback instead of LED.
    """
    
    def __init__(self, packet_queue: mp.Queue, status_queue: mp.Queue, inf_control_queue: mp.Queue):
        self.packet_queue = packet_queue
        self.status_queue = status_queue
        self.inf_control_queue = inf_control_queue
        self.ser: Optional[serial.Serial] = None
        self.button: Optional[Button] = None
        self.shutdown_button: Optional[Button] = None
        self.transfer_button: Optional[Button] = None
        self.running = False
        self.last_button_press = 0  # For debouncing
        self.recording = False
        self.draining = False  # Flag to indicate pipeline draining in progress
        self.drain_start_time = 0  # When the drain started
        self.countdown_active = False  # Flag to prevent multiple countdowns
        self.session_id = 0  # Increments with each recording to mark packets
        self.exiting = False
        self.transfer_thread = None  # Track active transfer thread
    
    def setup(self):
        """Initialize hardware connections"""
        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2.5)  # Wait for USB connection and Pico boot
        
        # Check for READY message from Pico
        bytes_waiting = self.ser.in_waiting
        self.status_queue.put(('DEBUG', f'After boot wait: {bytes_waiting} bytes in buffer'))
        
        if bytes_waiting > 0:
            try:
                ready_msg = self.ser.readline().decode().strip()
                self.status_queue.put(('INFO', f'Pico boot: {ready_msg}'))
            except Exception as e:
                self.status_queue.put(('WARN', f'Boot message error: {e}'))
        
        self.button = Button(BUTTON_PIN)
        self.button.when_pressed = self.toggle_recording
        
        self.shutdown_button = Button(SHUTDOWN_BUTTON_PIN)
        self.shutdown_button.when_pressed = self.initiate_shutdown
        
        self.transfer_button = Button(TRANSFER_BUTTON_PIN)
        self.transfer_button.when_pressed = self.initiate_transfer
        
        self.status_queue.put(('INFO', 'Acquisition process initialized'))
    
    def _find_marker(self) -> bool:
        """Sync recovery: scan for MARKER"""
        buffer = bytearray()
        while len(buffer) < len(MARKER):
            byte = self.ser.read(1)
            if not byte:
                return False
            buffer.append(byte[0])
        
        while True:
            if bytes(buffer[-len(MARKER):]) == MARKER:
                return True
            
            byte = self.ser.read(1)
            if not byte:
                return False
            buffer.append(byte[0])
            
            if len(buffer) > 1000:
                return False
    
    def _read_packet(self) -> Optional[bytes]:
        """Read one binary packet with MARKER sync"""
        # Read MARKER
        marker = self.ser.read(len(MARKER))
        
        if marker != MARKER:
            # Lost sync - show what we got instead
            if len(marker) == len(MARKER):
                hex_str = marker.hex()
                self.status_queue.put(('WARN', f'Sync lost - got {hex_str} instead of MARKER'))
            else:
                self.status_queue.put(('WARN', f'Sync lost - incomplete marker ({len(marker)} bytes)'))
            
            if self._find_marker():
                return self._read_packet()
            else:
                return None
        
        # Read data
        data = self.ser.read(PACKET_DATA_SIZE)
        
        if len(data) != PACKET_DATA_SIZE:
            self.status_queue.put(('WARN', f'Incomplete packet: {len(data)}/{PACKET_DATA_SIZE} bytes'))
            return None
        
        return data
    
    def start_acquisition(self):
        """Start ECG acquisition"""
        if self.recording:
            return
        
        self.recording = True
        self.session_id += 1  # New session ID for this recording
        
        # Clear any stale data from previous session
        # 1. Clear serial buffer
        bytes_before = self.ser.in_waiting
        self.ser.reset_input_buffer()
        if bytes_before > 0:
            self.status_queue.put(('DEBUG', f'Cleared {bytes_before} bytes before START'))
        
        # 2. Drain packet queue (stale packets from previous recording)
        packets_cleared = 0
        while not self.packet_queue.empty():
            try:
                self.packet_queue.get_nowait()
                packets_cleared += 1
            except:
                break
        if packets_cleared > 0:
            self.status_queue.put(('DEBUG', f'Cleared {packets_cleared} stale packets from queue'))
        
        # 3. Send session ID to inference so it knows to start fresh
        self.inf_control_queue.put(f'NEW_SESSION:{self.session_id}')
        
        # Send START command to Pico
        self.ser.write(b"START\n")
        self.ser.flush()
        
        # Wait for ACK with timeout
        time.sleep(0.1)
        
        if self.ser.in_waiting > 0:
            try:
                response = self.ser.readline().decode().strip()
                self.status_queue.put(('INFO', f'Pico ACK: {response}'))
                
                # If Pico was already recording (e.g. after unclean shutdown),
                # force a stop+start cycle so sample_count resets to 0
                if response == 'ALREADY_RECORDING':
                    self.status_queue.put(('WARN', 'Pico was still recording — resetting'))
                    self.ser.write(b"STOP\n")
                    self.ser.flush()
                    time.sleep(0.15)
                    self.ser.reset_input_buffer()
                    self.ser.write(b"START\n")
                    self.ser.flush()
                    time.sleep(0.1)
                    if self.ser.in_waiting > 0:
                        try:
                            self.ser.readline()  # consume ACK
                        except:
                            pass
            except UnicodeDecodeError as e:
                self.status_queue.put(('WARN', f'Binary data in ACK: {e}'))
        else:
            self.status_queue.put(('WARN', 'No ACK received from Pico'))
        
        # Give timer a moment to start
        time.sleep(0.05)
        
        self.status_queue.put(('STATUS', 'RECORDING_START'))
    
    def stop_acquisition(self):
        """Stop ECG acquisition"""
        if not self.recording:
            return
        
        self.recording = False
        
        # Send STOP command
        try:
            self.ser.write(b"STOP\n")
            self.ser.flush()
        except:
            pass
        
        self.status_queue.put(('STATUS', 'RECORDING_STOP'))
        
        # Start pipeline drain (non-blocking - handled in main loop)
        self.draining = True
        self.drain_start_time = time.time()
    
    def toggle_recording(self):
        """Button callback: toggle recording state"""
        if self.exiting or self.draining or self.countdown_active:
            return
        
        # Debounce: ignore if pressed too soon after last press
        current_time = time.time()
        if (current_time - self.last_button_press) < BUTTON_DEBOUNCE_TIME:
            return
        self.last_button_press = current_time
        
        if not self.recording:
            # Request countdown (handled in main loop to avoid blocking)
            self.countdown_active = True
        else:
            self.stop_acquisition()
    
    def initiate_shutdown(self):
        """Button callback: initiate system shutdown"""
        if self.exiting:
            return
        
        # Stop recording first so Pico resets cleanly on next boot
        if self.recording:
            self.stop_acquisition()
        
        # Show shutdown message on OLED
        self.status_queue.put(('STATUS', 'SHUTTING_DOWN'))
        time.sleep(1)  # Reduced from 2s - give OLED time to display message
        
        # Clear OLED display before shutdown
        self.status_queue.put(('STATUS', 'CLEAR_DISPLAY'))
        time.sleep(0.2)  # Reduced from 0.5s - brief time for OLED to clear
        
        # Execute shutdown command
        try:
            subprocess.run(['sudo', 'shutdown', 'now'], check=False, timeout=5)
        except Exception as e:
            self.status_queue.put(('ERROR', f'Shutdown failed: {e}'))
    
    def initiate_transfer(self):
        """Button callback: transfer data to USB device (runs in background thread)"""
        if self.exiting:
            return
        
        # Check if transfer already in progress
        if self.transfer_thread and self.transfer_thread.is_alive():
            self.status_queue.put(('WARN', 'Transfer already in progress'))
            return
        
        # Start transfer in background thread so buttons remain responsive
        self.transfer_thread = threading.Thread(target=self._do_transfer, daemon=True)
        self.transfer_thread.start()
    
    def _do_transfer(self):
        """Actual USB transfer implementation (runs in background thread)"""
        
        import subprocess
        import os
        from pathlib import Path
        
        # Show transfer starting message
        self.status_queue.put(('STATUS', 'USB_TRANSFER_START'))
        
        try:
            # Source directory - use absolute path to avoid home dir resolution issues
            source_dir = Path('/home/pi/thesis/data')
            self.status_queue.put(('INFO', f'Source dir: {source_dir} (exists={source_dir.exists()})'))

            # First, check if USB device is mounted
            result = subprocess.run(['lsblk', '-o', 'MOUNTPOINT,NAME,FSTYPE', '-n'], 
                                    capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            
            # Look for already mounted USB device
            usb_mount = None
            usb_device = None
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    mount = parts[0] if parts[0] else None
                    device = parts[1] if len(parts) > 1 else None
                    
                    # Check if mounted and in /media/
                    if mount and '/media/' in mount:
                        usb_mount = Path(mount)
                        break
            
            # If not mounted, try to find and mount USB device
            if not usb_mount:
                self.status_queue.put(('INFO', 'Looking for USB device...'))
                
                # Get list of block devices
                result = subprocess.run(['lsblk', '-o', 'NAME,TYPE,FSTYPE,SIZE', '-n'], 
                                        capture_output=True, text=True, check=True)
                lines = result.stdout.strip().split('\n')
                
                # Find USB partitions (typically sda1, sdb1, etc.)
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 3:
                        # Remove tree-drawing characters (└─, ├─, │, etc.)
                        name = parts[0].strip().lstrip('└├│─ ')
                        dev_type = parts[1]
                        fstype = parts[2] if len(parts) > 2 else ''
                        
                        # Look for partition with filesystem
                        if dev_type == 'part' and fstype and ('sd' in name or 'nvme' in name):
                            usb_device = f'/dev/{name}'
                            break
                
                if usb_device:
                    # Create mount point in /tmp (writable without sudo)
                    mount_point = Path('/tmp/usb_transfer')
                    
                    # Remove existing mount point if it exists
                    if mount_point.exists():
                        try:
                            subprocess.run(['sudo', 'umount', str(mount_point)], 
                                          check=False, timeout=3, capture_output=True)
                        except:
                            pass
                    
                    mount_point.mkdir(parents=True, exist_ok=True)
                    
                    self.status_queue.put(('INFO', f'Mounting {usb_device}...'))
                    
                    # Mount the device with proper permissions
                    try:
                        # Get current user uid and gid
                        import pwd
                        user_info = pwd.getpwnam('pi')
                        uid = user_info.pw_uid
                        gid = user_info.pw_gid
                        
                        # Mount with user ownership
                        subprocess.run(['sudo', 'mount', '-o', f'uid={uid},gid={gid}', 
                                       usb_device, str(mount_point)], 
                                      check=True, timeout=8, capture_output=True)
                        usb_mount = mount_point
                        self.status_queue.put(('INFO', 'USB mounted successfully'))
                    except subprocess.TimeoutExpired:
                        self.status_queue.put(('ERROR', 'Mount timeout - USB device may be faulty'))
                        self.status_queue.put(('STATUS', 'USB_TRANSFER_FAILED'))
                        return
                    except subprocess.CalledProcessError as e:
                        self.status_queue.put(('ERROR', f'Mount failed: {e}'))
                        self.status_queue.put(('STATUS', 'USB_TRANSFER_FAILED'))
                        return
            
            if not usb_mount:
                self.status_queue.put(('ERROR', 'No USB device found'))
                self.status_queue.put(('STATUS', 'USB_TRANSFER_FAILED'))
                return
            
            # Create destination directory
            dest_dir = usb_mount / 'ecg_data'
            try:
                dest_dir.mkdir(exist_ok=True)
            except Exception as e:
                self.status_queue.put(('WARN', f'Dest dir error: {e}'))
            
            # Count files to transfer
            files = list(source_dir.glob('*.csv'))
            file_count = len(files)
            
            if file_count == 0:
                self.status_queue.put(('WARN', f'No data files found in {source_dir}'))
                self.status_queue.put(('STATUS', 'USB_TRANSFER_COMPLETE'))
                return
            
            self.status_queue.put(('INFO', f'Transferring {file_count} files...'))
            
            # Copy files using rsync with update flag (skip existing newer files)
            # --update = skip files that are newer on receiver
            # --ignore-existing = skip files that already exist on receiver
            try:
                # Use --no-perms to avoid permission errors
                subprocess.run(['rsync', '-av', '--ignore-existing', '--no-perms',
                               f'{source_dir}/', f'{dest_dir}/'],
                              check=True, timeout=90, capture_output=True)
            except subprocess.TimeoutExpired:
                self.status_queue.put(('ERROR', 'Transfer timeout - too many files or slow USB'))
                self.status_queue.put(('STATUS', 'USB_TRANSFER_FAILED'))
                return
            except subprocess.CalledProcessError as e:
                # rsync exit code 23 = partial transfer (some files had errors but others succeeded)
                # This is often OK - usually means some files already exist or minor issues
                if e.returncode == 23:
                    self.status_queue.put(('WARN', 'Partial transfer - some files skipped'))
                else:
                    self.status_queue.put(('ERROR', f'Transfer failed with code {e.returncode}'))
                    self.status_queue.put(('STATUS', 'USB_TRANSFER_FAILED'))
                    return
            
            # Sync filesystem to ensure write completion
            try:
                subprocess.run(['sync'], check=True, timeout=10)
            except:
                pass  # sync rarely fails
            
            self.status_queue.put(('INFO', f'Transferred {file_count} files'))
            
            # Safely unmount the USB device
            try:
                subprocess.run(['sudo', 'umount', str(usb_mount)], 
                              check=True, timeout=8, capture_output=True)
                self.status_queue.put(('INFO', 'USB safely ejected'))
            except subprocess.TimeoutExpired:
                self.status_queue.put(('WARN', 'Unmount timeout - remove USB manually'))
            except:
                self.status_queue.put(('WARN', 'Unmount failed - remove USB manually'))
            
            self.status_queue.put(('STATUS', f'USB_TRANSFER_COMPLETE:{file_count}'))
            
        except subprocess.TimeoutExpired as e:
            self.status_queue.put(('ERROR', f'Transfer timeout: {e.cmd[0] if e.cmd else "unknown"}'))
            self.status_queue.put(('STATUS', 'USB_TRANSFER_FAILED'))
        except subprocess.CalledProcessError as e:
            self.status_queue.put(('ERROR', f'Transfer failed: {e}'))
            self.status_queue.put(('STATUS', 'USB_TRANSFER_FAILED'))
        except Exception as e:
            self.status_queue.put(('ERROR', f'Transfer error: {e}'))
            self.status_queue.put(('STATUS', 'USB_TRANSFER_FAILED'))
    
    def acquisition_loop(self):
        """Main acquisition loop - read packets and forward to segmentation"""
        packet_count = 0
        last_status_time = time.time()
        last_debug_time = time.time()
        last_packet_time = None
        expected_packet_interval = BUFFER_SIZE / 360.0  # Expected time between packets (256 samples @ 360 Hz = 0.711s)
        
        DRAIN_DURATION = 3.0  # Seconds to wait for pipeline drain
        countdown_start_time = 0
        countdown_value = 0

        # Packet-timeout diagnostic state
        _in_timeout = False          # True once timeout threshold is crossed
        _timeout_start = 0.0         # Wall time when timeout began
        _timeout_last_report = 0.0   # Wall time of last periodic update
        _bytes_at_timeout = 0        # ser.in_waiting snapshot on entry
        _TIMEOUT_REPORT_INTERVAL = 5.0  # Log a heartbeat every N seconds while stalled

        while self.running:
            # Handle countdown (non-blocking)
            if self.countdown_active:
                if countdown_value == 0:
                    # Start countdown
                    countdown_value = RECORDING_COUNTDOWN_SEC
                    countdown_start_time = time.time()
                    self.status_queue.put(('STATUS', f'COUNTDOWN:{countdown_value}'))
                else:
                    # Check if 1 second elapsed
                    elapsed = time.time() - countdown_start_time
                    if elapsed >= 1.0:
                        countdown_value -= 1
                        countdown_start_time = time.time()
                        
                        if countdown_value > 0:
                            self.status_queue.put(('STATUS', f'COUNTDOWN:{countdown_value}'))
                        else:
                            # Countdown complete - start recording
                            self.start_acquisition()
                            self.countdown_active = False
                            countdown_value = 0
            
            # Check if pipeline drain is complete
            if self.draining:
                if time.time() - self.drain_start_time >= DRAIN_DURATION:
                    # Drain complete - send stop signal
                    self.inf_control_queue.put('STOP_SESSION')
                    self.draining = False
            
            if self.recording:
                # Show serial buffer status every 2 seconds to debug
                now = time.time()
                if now - last_debug_time >= 2.0:
                    bytes_waiting = self.ser.in_waiting
                    self.status_queue.put(('DEBUG', f'Serial buffer: {bytes_waiting} bytes, Packets: {packet_count}'))
                    last_debug_time = now
                
                # Check for packet reception timeout
                if last_packet_time and (now - last_packet_time) > expected_packet_interval * 3:
                    gap = now - last_packet_time

                    if not _in_timeout:
                        # First detection: snapshot diagnostics once
                        _in_timeout = True
                        _timeout_start = now
                        _timeout_last_report = now
                        _bytes_at_timeout = self.ser.in_waiting
                        self.status_queue.put(('WARN',
                            f'Packet timeout: no packet for {gap:.2f}s '
                            f'(expected {expected_packet_interval:.2f}s) — '
                            f'serial buffer={_bytes_at_timeout}B, '
                            f'port_open={self.ser.isOpen()}, '
                            f'packets_so_far={packet_count}'))
                    elif (now - _timeout_last_report) >= _TIMEOUT_REPORT_INTERVAL:
                        # Periodic heartbeat while still stalled
                        buf = self.ser.in_waiting
                        growing = '↑' if buf > _bytes_at_timeout else ('=' if buf == _bytes_at_timeout else '↓')
                        self.status_queue.put(('WARN',
                            f'Still stalled: {gap:.1f}s elapsed, '
                            f'serial buffer={buf}B {growing}, '
                            f'port_open={self.ser.isOpen()}'))
                        _timeout_last_report = now
                        _bytes_at_timeout = buf
                
                # Check if full packet is available
                if self.ser.in_waiting >= PACKET_SIZE:
                    packet_data = self._read_packet()
                    
                    if packet_data:
                        # Forward to segmentation process with arrival timestamp and session ID
                        arrival_time = time.time()
                        self.packet_queue.put((packet_data, arrival_time, self.session_id))
                        packet_count += 1

                        # Recovery log: clear timeout state and report gap length
                        if _in_timeout:
                            gap_duration = arrival_time - _timeout_start
                            self.status_queue.put(('WARN',
                                f'Packet stream recovered after {gap_duration:.2f}s gap '
                                f'(port_open={self.ser.isOpen()})'))
                            _in_timeout = False
                            _timeout_start = 0.0
                        
                        # First packet notification
                        if packet_count == 1:
                            self.status_queue.put(('INFO', 'First packet received!'))
                        
                        # Validate packet timing
                        if last_packet_time:
                            interval = arrival_time - last_packet_time
                            if interval > expected_packet_interval * 2:
                                self.status_queue.put(('WARN', 
                                    f'Late packet: {interval:.3f}s (expected {expected_packet_interval:.3f}s)'))
                        
                        last_packet_time = arrival_time
                        
                        # Status update every 10 packets
                        if now - last_status_time >= 5.0:
                            self.status_queue.put(('DEBUG', f'Packets received: {packet_count}'))
                            last_status_time = now
                else:
                    time.sleep(0.001)  # 1ms wait for data
            else:
                time.sleep(0.01)  # 10ms when not recording
                packet_count = 0  # Reset on stop
                last_packet_time = None  # Reset timing tracker
    
    def run(self):
        """Process main entry point"""
        self.setup()
        self.running = True
        
        try:
            self.acquisition_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.exiting = True
        if self.recording:
            self.stop_acquisition()
        
        # Close GPIO buttons to release pins
        if self.button:
            self.button.close()
        if self.shutdown_button:
            self.shutdown_button.close()
        if self.transfer_button:
            self.transfer_button.close()
        
        if self.ser and self.ser.is_open:
            self.ser.close()
        
        self.status_queue.put(('INFO', 'Acquisition process terminated'))


def acquisition_worker(packet_queue: mp.Queue, status_queue: mp.Queue, control_queue: mp.Queue, inf_control_queue: mp.Queue):
    """
    Worker function for acquisition process.
    
    Args:
        packet_queue: Output queue for binary packets
        status_queue: Output queue for status messages
        control_queue: Input queue for control commands
        inf_control_queue: Output queue to send control messages to inference process
    """
    process = AcquisitionProcess(packet_queue, status_queue, inf_control_queue)
    process.run()
