#!/usr/bin/env python3
"""
OLED Display Process - 1.3" OLED display (SH1106) for status feedback.
- Press to Start
- Recording in Progress
- Rhythm classification results (NSR, AFIB, PVC, LBBB)
- Error messages
"""

import time
import multiprocessing as mp
from PIL import Image, ImageDraw, ImageFont

class OLEDDisplay:
    """OLED display controller for ECG pipeline status."""
    
    def __init__(self, i2c_address: int = 0x3C, i2c_bus: int = 1):
        self.i2c_address = i2c_address
        self.i2c_bus = i2c_bus
        self.device = None
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.running = False
        self.current_state = "IDLE"
        self.current_rhythm = None
        
    def setup(self):
        """Initialize OLED display and fonts."""
        from luma.core.interface.serial import i2c
        from luma.oled.device import sh1106, ssd1306
        
        try:
            import subprocess
            subprocess.run(['gpio', 'mode', '8', 'ALT0'], check=False, capture_output=True)
            subprocess.run(['gpio', 'mode', '9', 'ALT0'], check=False, capture_output=True)
            time.sleep(0.1)
        except Exception as e:
            pass
        
        for attempt in range(2):
            if attempt == 1:
                time.sleep(0.2)
            
            try:
                serial = i2c(port=self.i2c_bus, address=self.i2c_address)
                device = sh1106(serial, width=128, height=64)
                device.contrast(0x7F)
                time.sleep(0.1)
                
                test_img = Image.new("1", (128, 64), color=0)
                draw = ImageDraw.Draw(test_img)
                draw.rectangle((0, 0, 127, 63), outline=1)
                device.display(test_img)
                device.clear()
                device.show()
                
                self.device = device
                break
            except Exception as e:
                if attempt == 1:
                    try:
                        serial = i2c(port=self.i2c_bus, address=self.i2c_address)
                        device = ssd1306(serial, width=128, height=64)
                        device.contrast(0x7F)
                        time.sleep(0.1)
                        
                        test_img = Image.new("1", (128, 64), color=0)
                        draw = ImageDraw.Draw(test_img)
                        draw.rectangle((0, 0, 127, 63), outline=1)
                        device.display(test_img)
                        device.clear()
                        device.show()
                        
                        self.device = device
                        break
                    except:
                        raise RuntimeError(f"OLED init failed at 0x{self.i2c_address:02x}")
        
        if self.device is None:
            raise RuntimeError("OLED device not initialized")
        
        self._load_fonts()
        
        self.show_message("ECG", "READY", size="large")
        time.sleep(1)
    
    def _load_fonts(self):
        """Load TrueType fonts with fallbacks."""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        
        font_path = None
        for path in font_paths:
            try:
                ImageFont.truetype(path, 10)
                font_path = path
                break
            except:
                continue
        
        if font_path:
            self.font_large = ImageFont.truetype(font_path, 28)
            self.font_medium = ImageFont.truetype(font_path, 16)
            self.font_small = ImageFont.truetype(font_path, 12)
        else:
            self.font_large = ImageFont.load_default()
            self.font_medium = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
    
    def _measure_text(self, draw: ImageDraw.ImageDraw, text: str, font) -> tuple:
        """Measure text dimensions."""
        if hasattr(draw, "textbbox"):
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
            return (r - l, b - t)
        elif hasattr(draw, "textsize"):
            return draw.textsize(text, font=font)
        else:
            try:
                return font.getsize(text)
            except:
                return (len(text) * 8, 10)
    
    def show_message(self, line1: str, line2: str = "", size: str = "medium"):
        """Display a two-line message on OLED."""
        if not self.device:
            return
        
        if size == "large":
            font = self.font_large
        elif size == "small":
            font = self.font_small
        else:
            font = self.font_medium
        
        img = Image.new("1", (128, 64), color=0)
        draw = ImageDraw.Draw(img)
        
        w1, h1 = self._measure_text(draw, line1, font)
        x1 = (128 - w1) // 2
        
        if line2:
            y1 = (64 - h1) // 2 - 8
            draw.text((x1, y1), line1, font=font, fill=255)
            
            w2, h2 = self._measure_text(draw, line2, font)
            x2 = (128 - w2) // 2
            y2 = (64 - h2) // 2 + 8
            draw.text((x2, y2), line2, font=font, fill=255)
        else:
            y1 = (64 - h1) // 2
            draw.text((x1, y1), line1, font=font, fill=255)
        
        self.device.display(img)
    
    def show_rhythm(self, rhythm: str, duration: int = 0):
        """Display rhythm classification with recording duration."""
        if not self.device:
            return
        
        self.current_rhythm = rhythm
        
        rhythm_map = {
            'NSR': 'NORMAL',
            'AFIB': 'A-FIB',
            'PVC': 'PVC',
            'LBBB': 'LBBB',
            'NORMAL': 'NORMAL'
        }
        
        display_text = rhythm_map.get(rhythm, rhythm)
        
        img = Image.new("1", (128, 64), color=0)
        draw = ImageDraw.Draw(img)
        
        if duration > 0:
            w, h = self._measure_text(draw, display_text, self.font_large)
            x = (128 - w) // 2
            y = (64 - h) // 2 - 10
            draw.text((x, y), display_text, font=self.font_large, fill=255)
            
            mins = duration // 60
            secs = duration % 60
            duration_text = f"{mins:02d}:{secs:02d}"
            
            w2, h2 = self._measure_text(draw, duration_text, self.font_medium)
            x2 = (128 - w2) // 2
            y2 = (64 - h2) // 2 + 12
            draw.text((x2, y2), duration_text, font=self.font_medium, fill=255)
        else:
            w, h = self._measure_text(draw, display_text, self.font_large)
            x = (128 - w) // 2
            y = (64 - h) // 2
            draw.text((x, y), display_text, font=self.font_large, fill=255)
        
        self.device.display(img)
    
    def show_idle(self):
        """Show idle state - ready to record with button menu."""
        self.current_state = "IDLE"
        
        img = Image.new("1", (128, 64), color=0)
        draw = ImageDraw.Draw(img)
        
        title = "MENU"
        w, h = self._measure_text(draw, title, self.font_medium)
        x = (128 - w) // 2
        draw.text((x, 2), title, font=self.font_medium, fill=255)
        
        options = [
            "1: Record",
            "2: Transfer",
            "3: Shutdown"
        ]
        
        y_start = 22
        line_height = 14
        
        for i, option in enumerate(options):
            y = y_start + (i * line_height)
            draw.text((10, y), option, font=self.font_small, fill=255)
        
        self.device.display(img)
    
    def show_recording(self):
        """Show recording in progress."""
        self.current_state = "RECORDING"
        self.show_message("RECORDING", size="medium")
    
    def show_processing(self):
        """Show processing state."""
        self.current_state = "PROCESSING"
        self.show_message("ANALYZING", "ECG DATA", size="medium")
    
    def show_error(self, error_msg: str):
        """Show error message."""
        self.current_state = "ERROR"
        self.show_message("ERROR", error_msg, size="small")
    
    def update_loop(self, status_queue: mp.Queue, control_queue: mp.Queue):
        """Main update loop - listens for status messages and updates display."""
        self.running = True
        self.show_idle()
        
        while self.running:
            try:
                msg_type, msg_content = status_queue.get(timeout=0.1)
                
                if msg_type == 'STATUS':
                    if msg_content == 'RECORDING_START':
                        self.show_recording()
                    elif msg_content == 'RECORDING_STOP':
                        self.show_idle()
                    elif msg_content.startswith('SAVED:'):
                        parts = msg_content.split(':', 2)
                        if len(parts) >= 3:
                            full_filename = parts[1].strip()
                            duration = int(parts[2].strip())
                            
                            if full_filename.startswith('ecg_'):
                                compact = full_filename[4:].replace('_', '-')
                            else:
                                compact = full_filename[-15:]
                            
                            mins = duration // 60
                            secs = duration % 60
                            duration_text = f"{mins:02d}:{secs:02d}"
                            
                            self.show_message(compact, duration_text, size="small")
                        else:
                            full_filename = parts[1].strip() if len(parts) > 1 else "unknown"
                            if full_filename.startswith('ecg_'):
                                compact = full_filename[4:].replace('_', '-')
                            else:
                                compact = full_filename[-15:]
                            self.show_message("SAVED", compact, size="small")
                        
                        time.sleep(3)
                        self.show_idle()
                    elif msg_content.startswith('RHYTHM:'):
                        parts = msg_content.split(':', 2)
                        rhythm = parts[1].strip()
                        duration = int(parts[2].strip()) if len(parts) > 2 else 0
                        self.show_rhythm(rhythm, duration)
                    elif msg_content == 'SHUTTING_DOWN':
                        self.show_message("SHUTTING", "DOWN...", size="medium")
                    elif msg_content == 'USB_TRANSFER_START':
                        self.show_message("USB", "TRANSFER...", size="medium")
                    elif msg_content.startswith('USB_TRANSFER_COMPLETE'):
                        parts = msg_content.split(':', 1)
                        if len(parts) > 1:
                            count = parts[1].strip()
                            self.show_message("TRANSFER OK", f"{count} files", size="medium")
                        else:
                            self.show_message("TRANSFER", "COMPLETE", size="medium")
                        time.sleep(3)
                        self.show_idle()
                    elif msg_content == 'USB_TRANSFER_FAILED':
                        self.show_message("TRANSFER", "FAILED", size="medium")
                        time.sleep(3)
                        self.show_idle()
                    elif msg_content == 'CLEAR_DISPLAY':
                        if self.device:
                            self.device.clear()
                            self.device.show()
                
                elif msg_type == 'ERROR':
                    self.show_error(msg_content[:20])
                
            except mp.queues.Empty:
                continue
            except KeyboardInterrupt:
                break
    
    def cleanup(self):
        """Clean up display."""
        if self.device:
            try:
                self.device.clear()
                self.device.show()
            except:
                pass


def oled_worker(status_queue: mp.Queue, control_queue: mp.Queue):
    """Worker function for OLED display process."""
    display = OLEDDisplay()
    
    try:
        display.setup()
        display.update_loop(status_queue, control_queue)
    except Exception as e:
        print(f"[OLED] Error: {e}")
    finally:
        display.cleanup()
