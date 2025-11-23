#!/usr/bin/env python3
"""
Acquisition Process - Real-time ECG data acquisition with GPIO control.
- USB serial acquisition (MARKER-based binary packets)
- Button/LED control
- Forward packets to segmentation process
"""

import serial
import time
import struct
import multiprocessing as mp
from gpiozero import Button
from typing import Optional

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
BUFFER_SIZE = 256
MARKER = b"MARKER"
PACKET_DATA_SIZE = BUFFER_SIZE * 6
PACKET_SIZE = len(MARKER) + PACKET_DATA_SIZE

BUTTON_PIN = 26
SHUTDOWN_BUTTON_PIN = 13
TRANSFER_BUTTON_PIN = 19

class AcquisitionProcess:
    """Acquisition + GPIO control process."""
    
    def __init__(self, packet_queue: mp.Queue, status_queue: mp.Queue, inf_control_queue: mp.Queue):
        self.packet_queue = packet_queue
        self.status_queue = status_queue
        self.inf_control_queue = inf_control_queue
        self.ser: Optional[serial.Serial] = None
        self.button: Optional[Button] = None
        self.shutdown_button: Optional[Button] = None
        self.transfer_button: Optional[Button] = None
        self.running = False
        self.recording = False
        self.exiting = False
    
    def setup(self):
        """Initialize hardware connections."""
        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2.5)
        
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
        """Scan for MARKER to recover synchronization."""
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
        """Read one binary packet with MARKER synchronization."""
        marker = self.ser.read(len(MARKER))
        
        if marker != MARKER:
            if len(marker) == len(MARKER):
                hex_str = marker.hex()
                self.status_queue.put(('WARN', f'Sync lost - got {hex_str} instead of MARKER'))
            else:
                self.status_queue.put(('WARN', f'Sync lost - incomplete marker ({len(marker)} bytes)'))
            
            if self._find_marker():
                return self._read_packet()
            else:
                return None
        
        data = self.ser.read(PACKET_DATA_SIZE)
        
        if len(data) != PACKET_DATA_SIZE:
            self.status_queue.put(('WARN', f'Incomplete packet: {len(data)}/{PACKET_DATA_SIZE} bytes'))
            return None
        
        return data
    
    def start_acquisition(self):
        """Start ECG acquisition."""
        if self.recording:
            return
        
        self.recording = True
        
        bytes_before = self.ser.in_waiting
        self.ser.reset_input_buffer()
        if bytes_before > 0:
            self.status_queue.put(('DEBUG', f'Cleared {bytes_before} bytes before START'))
        
        self.ser.write(b"START\n")
        self.ser.flush()
        
        time.sleep(0.1)
        
        if self.ser.in_waiting > 0:
            try:
                response = self.ser.readline().decode().strip()
                self.status_queue.put(('INFO', f'Pico ACK: {response}'))
            except UnicodeDecodeError as e:
                self.status_queue.put(('WARN', f'Binary data in ACK: {e}'))
        else:
            self.status_queue.put(('WARN', 'No ACK received from Pico'))
        
        time.sleep(0.05)
        
        self.status_queue.put(('STATUS', 'RECORDING_START'))
    
    def stop_acquisition(self):
        """Stop ECG acquisition."""
        if not self.recording:
            return
        
        self.recording = False
        
        try:
            self.ser.write(b"STOP\n")
            self.ser.flush()
        except:
            pass
        
        self.status_queue.put(('STATUS', 'RECORDING_STOP'))
        self.inf_control_queue.put('STOP_SESSION')
    
    def toggle_recording(self):
        """Button callback: toggle recording state."""
        if self.exiting:
            return
        
        if not self.recording:
            self.start_acquisition()
        else:
            self.stop_acquisition()
    
    def initiate_shutdown(self):
        """Button callback: initiate system shutdown."""
        if self.exiting:
            return
        
        import subprocess
        
        self.status_queue.put(('STATUS', 'SHUTTING_DOWN'))
        time.sleep(2)
        
        self.status_queue.put(('STATUS', 'CLEAR_DISPLAY'))
        time.sleep(0.5)
        
        try:
            subprocess.run(['sudo', 'shutdown', 'now'], check=False)
        except Exception as e:
            self.status_queue.put(('ERROR', f'Shutdown failed: {e}'))
    
    def initiate_transfer(self):
        """Transfer data to USB device."""
        if self.exiting:
            return
        
        import subprocess
        import os
        from pathlib import Path
        
        self.status_queue.put(('STATUS', 'USB_TRANSFER_START'))
        
        try:
            source_dir = Path.home() / 'thesis' / 'scripts' / 'system' / 'deployment_data'
            
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
                    
                    if mount and '/media/' in mount:
                        usb_mount = Path(mount)
                        break
            
            if not usb_mount:
                self.status_queue.put(('INFO', 'Looking for USB device...'))
                
                result = subprocess.run(['lsblk', '-o', 'NAME,TYPE,FSTYPE,SIZE', '-n'], 
                                        capture_output=True, text=True, check=True)
                lines = result.stdout.strip().split('\n')
                
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 3:
                        name = parts[0].strip().lstrip('└├│─ ')
                        dev_type = parts[1]
                        fstype = parts[2] if len(parts) > 2 else ''
                        
                        if dev_type == 'part' and fstype and ('sd' in name or 'nvme' in name):
                            usb_device = f'/dev/{name}'
                            break
                
                if usb_device:
                    mount_point = Path('/tmp/usb_transfer')
                    
                    if mount_point.exists():
                        try:
                            subprocess.run(['sudo', 'umount', str(mount_point)], 
                                          check=False, timeout=2)
                        except:
                            pass
                    
                    mount_point.mkdir(parents=True, exist_ok=True)
                    
                    self.status_queue.put(('INFO', f'Mounting {usb_device}...'))
                    
                    try:
                        import pwd
                        user_info = pwd.getpwnam('pi')
                        uid = user_info.pw_uid
                        gid = user_info.pw_gid
                        
                        subprocess.run(['sudo', 'mount', '-o', f'uid={uid},gid={gid}', 
                                       usb_device, str(mount_point)], 
                                      check=True, timeout=5)
                        usb_mount = mount_point
                        self.status_queue.put(('INFO', 'USB mounted successfully'))
                    except subprocess.CalledProcessError as e:
                        self.status_queue.put(('ERROR', f'Mount failed: {e}'))
                        self.status_queue.put(('STATUS', 'USB_TRANSFER_FAILED'))
                        return
            
            if not usb_mount:
                self.status_queue.put(('ERROR', 'No USB device found'))
                self.status_queue.put(('STATUS', 'USB_TRANSFER_FAILED'))
                return
            
            dest_dir = usb_mount / 'ecg_data'
            try:
                dest_dir.mkdir(exist_ok=True)
            except Exception as e:
                self.status_queue.put(('WARN', f'Dest dir error: {e}'))
            
            files = list(source_dir.glob('*.csv'))
            file_count = len(files)
            
            if file_count == 0:
                self.status_queue.put(('WARN', 'No data files to transfer'))
                self.status_queue.put(('STATUS', 'USB_TRANSFER_COMPLETE'))
                return
            
            self.status_queue.put(('INFO', f'Transferring {file_count} files...'))
            
            subprocess.run(['rsync', '-av', '--ignore-existing',
                           f'{source_dir}/', f'{dest_dir}/'],
                          check=True, timeout=60)
            
            subprocess.run(['sync'], check=True)
            
            self.status_queue.put(('INFO', f'Transferred {file_count} files'))
            
            try:
                subprocess.run(['sudo', 'umount', str(usb_mount)], check=True, timeout=5)
                self.status_queue.put(('INFO', 'USB safely ejected'))
            except:
                pass
            
            self.status_queue.put(('STATUS', f'USB_TRANSFER_COMPLETE:{file_count}'))
            
        except subprocess.CalledProcessError as e:
            self.status_queue.put(('ERROR', f'Transfer failed: {e}'))
            self.status_queue.put(('STATUS', 'USB_TRANSFER_FAILED'))
        except Exception as e:
            self.status_queue.put(('ERROR', f'Transfer error: {e}'))
            self.status_queue.put(('STATUS', 'USB_TRANSFER_FAILED'))
    
    def acquisition_loop(self):
        """Main acquisition loop."""
        packet_count = 0
        last_status_time = time.time()
        last_debug_time = time.time()
        
        while self.running:
            if self.recording:
                now = time.time()
                if now - last_debug_time >= 2.0:
                    bytes_waiting = self.ser.in_waiting
                    self.status_queue.put(('DEBUG', f'Serial buffer: {bytes_waiting} bytes, Packets: {packet_count}'))
                    last_debug_time = now
                
                if self.ser.in_waiting >= PACKET_SIZE:
                    packet_data = self._read_packet()
                    
                    if packet_data:
                        arrival_time = time.time()
                        self.packet_queue.put((packet_data, arrival_time))
                        packet_count += 1
                        
                        if packet_count == 1:
                            self.status_queue.put(('INFO', 'First packet received!'))
                        
                        if now - last_status_time >= 5.0:
                            self.status_queue.put(('DEBUG', f'Packets received: {packet_count}'))
                            last_status_time = now
                else:
                    time.sleep(0.001)
            else:
                time.sleep(0.01)
                packet_count = 0
    
    def run(self):
        """Process main entry point."""
        self.setup()
        self.running = True
        
        try:
            self.acquisition_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        self.exiting = True
        if self.recording:
            self.stop_acquisition()
        
        if self.ser and self.ser.is_open:
            self.ser.close()
        
        self.status_queue.put(('INFO', 'Acquisition process terminated'))


def acquisition_worker(packet_queue: mp.Queue, status_queue: mp.Queue, control_queue: mp.Queue, inf_control_queue: mp.Queue):
    """Worker function for acquisition process."""
    process = AcquisitionProcess(packet_queue, status_queue, inf_control_queue)
    process.run()
