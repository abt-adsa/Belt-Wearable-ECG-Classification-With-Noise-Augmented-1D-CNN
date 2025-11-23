"""
Pico ECG Acquisition - Production binary protocol.
- MARKER-based sync (6 bytes)
- 256-sample buffers (0.71s @ 360 Hz)
- 4 packets = 1024 samples (TFLite input size)

Packet: MARKER (6 bytes) + Data (1536 bytes: 256 × 6 bytes) = 1542 bytes
"""
import machine
import micropython
import time
import sys
import select
import struct

micropython.alloc_emergency_exception_buf(100)

SAMPLE_RATE_HZ = 360
ADC_PIN = 26
BUFFER_SIZE = 256
MARKER = b"MARKER"

led = machine.Pin("LED", machine.Pin.OUT)

class ECGAcquisition:
    def __init__(self):
        self.adc = machine.ADC(ADC_PIN)
        self.is_recording = False
        self.sample_count = 0
        self.adc_buffer = bytearray(BUFFER_SIZE * 6)
        self.buffer_index = 0
        self.start_time_ms = 0
        self.timer = machine.Timer()
        self.poll_obj = select.poll()
        self.poll_obj.register(sys.stdin, select.POLLIN)
    
    def start_recording(self):
        if not self.is_recording:
            led.on()
            self.is_recording = True
            self.sample_count = 0
            self.buffer_index = 0
            self.start_time_ms = time.ticks_ms()
            
            self.timer.init(
                mode=machine.Timer.PERIODIC,
                freq=SAMPLE_RATE_HZ,
                callback=self._sample_callback
            )
    
    def stop_recording(self):
        if self.is_recording:
            led.off()
            self.timer.deinit()
            self.is_recording = False
            
            if self.buffer_index > 0:
                self._flush_buffer()
    
    def _sample_callback(self, timer):
        """Timer callback - acquire and buffer samples."""
        if self.buffer_index < BUFFER_SIZE:
            adc_value = self.adc.read_u16()
            timestamp_ms = time.ticks_add(self.start_time_ms, 
                                         int((self.sample_count * 1000) / SAMPLE_RATE_HZ))
            
            offset = self.buffer_index * 6
            struct.pack_into('<IH', self.adc_buffer, offset, timestamp_ms, adc_value)
            
            self.buffer_index += 1
            self.sample_count += 1
            
            if self.buffer_index >= BUFFER_SIZE:
                self._flush_buffer()
    
    def _flush_buffer(self):
        """Send binary packet with MARKER sync."""
        data_size = self.buffer_index * 6
        
        try:
            sys.stdout.buffer.write(MARKER)
            sys.stdout.buffer.write(memoryview(self.adc_buffer)[0:data_size])
            sys.stdout.buffer.flush()
        except:
            pass
        
        self.buffer_index = 0
    
    def check_command(self):
        """Process commands from receiver."""
        if self.poll_obj.poll(0):
            try:
                line = sys.stdin.readline().strip()
                if line:
                    command = line.upper()
                    
                    if command == "START":
                        sys.stdout.write("ACK\n")
                        self.start_recording()
                    elif command == "STOP":
                        sys.stdout.write("ACK\n")
                        self.stop_recording()
                    elif command == "PING":
                        sys.stdout.write("PONG\n")
            except:
                pass
    
    def run(self):
        sys.stdout.write("READY\n")
        
        while True:
            self.check_command()
            time.sleep_ms(10)

if __name__ == "__main__":
    ecg = ECGAcquisition()
    ecg.run()
