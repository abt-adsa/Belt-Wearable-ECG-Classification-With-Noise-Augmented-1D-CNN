#!/usr/bin/env python3
"""
OLED Display Module for ECG Pipeline
=====================================
Manages 1.3" OLED display (SH1106) for status feedback:
- Press to Start
- Recording in Progress
- Rhythm classification results (NSR, AFIB, PVC, LBBB)
- Error messages

Replaces LED status indicators with visual feedback.
"""

import time
import multiprocessing as mp
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

class OLEDDisplay:
    """
    OLED display controller for ECG pipeline status.
    Runs in separate process to avoid blocking main pipeline.
    """
    
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
        self.current_duration = 0
        
    def setup(self):
        """Initialize OLED display and fonts"""
        from luma.core.interface.serial import i2c
        from luma.oled.device import sh1106, ssd1306
        
        # Force I2C pins back to ALT0 mode BEFORE any I2C operations
        # This prevents the issue where GPIO libraries change pin modes from I2C to GPIO
        try:
            import subprocess
            # SDA = GPIO2 = WiringPi pin 8
            subprocess.run(['gpio', 'mode', '8', 'ALT0'], check=False, capture_output=True)
            # SCL = GPIO3 = WiringPi pin 9
            subprocess.run(['gpio', 'mode', '9', 'ALT0'], check=False, capture_output=True)
            time.sleep(0.1)
        except Exception as e:
            # If gpio command not available, continue anyway
            pass
        
        # Try SH1106 first (1.3" displays), retry once if bus stuck
        for attempt in range(2):
            if attempt == 1:
                time.sleep(0.2)
            
            try:
                serial = i2c(port=self.i2c_bus, address=self.i2c_address)
                device = sh1106(serial, width=128, height=64)
                device.contrast(0x7F)
                time.sleep(0.1)
                
                # Verify with data transfer
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
                    # Try SSD1306 fallback
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
        
        # Load fonts
        self._load_fonts()
        
        # Show startup message (shorter text)
        self.show_message("ECG", "READY", size="large")
        time.sleep(1)
    
    def _load_fonts(self):
        """Load TrueType fonts with fallbacks"""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        
        font_path = None
        for path in font_paths:
            try:
                # Test if font exists
                ImageFont.truetype(path, 10)
                font_path = path
                break
            except:
                continue
        
        if font_path:
            # Smaller fonts for 1.3" display to fit text properly
            self.font_large = ImageFont.truetype(font_path, 28)   # For single-word rhythm
            self.font_medium = ImageFont.truetype(font_path, 16)  # For two-line messages
            self.font_small = ImageFont.truetype(font_path, 12)
        else:
            # Fallback to default fonts
            self.font_large = ImageFont.load_default()
            self.font_medium = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
    
    def _measure_text(self, draw: ImageDraw.ImageDraw, text: str, font) -> tuple:
        """Measure text dimensions"""
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
        """
        Display a two-line message on OLED.
        
        Args:
            line1: Top line text
            line2: Bottom line text (optional)
            size: Font size - "large", "medium", or "small"
        """
        if not self.device:
            return
        
        # Select font
        if size == "large":
            font = self.font_large
        elif size == "small":
            font = self.font_small
        else:
            font = self.font_medium
        
        # Create image
        img = Image.new("1", (128, 64), color=0)
        draw = ImageDraw.Draw(img)
        
        # Draw line 1 (centered)
        w1, h1 = self._measure_text(draw, line1, font)
        x1 = (128 - w1) // 2
        
        if line2:
            # Two lines - split vertically
            y1 = (64 - h1) // 2 - 8
            draw.text((x1, y1), line1, font=font, fill=255)
            
            # Draw line 2
            w2, h2 = self._measure_text(draw, line2, font)
            x2 = (128 - w2) // 2
            y2 = (64 - h2) // 2 + 8
            draw.text((x2, y2), line2, font=font, fill=255)
        else:
            # Single line - centered
            y1 = (64 - h1) // 2
            draw.text((x1, y1), line1, font=font, fill=255)
        
        self.device.display(img)
    
    def show_rhythm(self, rhythm: str, duration: int = 0):
        """
        Display rhythm classification with recording duration.
        
        Args:
            rhythm: One of "NSR", "AFIB", "PVC", "LBBB"
            duration: Recording duration in seconds (optional)
        """
        if not self.device:
            return
        
        # Remove mode indicator (if present) - for discrete operation
        rhythm = rhythm.replace('[M]', '').strip()
        
        self.current_rhythm = rhythm
        self.current_duration = duration
        
        # Map rhythm names to display-friendly versions
        rhythm_map = {
            'NSR': 'NORMAL',  # More user-friendly than "NSR"
            'AFIB': 'A-FIB',
            'PVC': 'PVC',
            'LBBB': 'LBBB',
            'NORMAL': 'NORMAL'
        }
        
        display_text = rhythm_map.get(rhythm, rhythm)
        
        # Create image
        img = Image.new("1", (128, 64), color=0)
        draw = ImageDraw.Draw(img)
        
        if duration > 0:
            # Two-line display: rhythm + duration
            w, h = self._measure_text(draw, display_text, self.font_large)
            x = (128 - w) // 2
            y = (64 - h) // 2 - 10
            draw.text((x, y), display_text, font=self.font_large, fill=255)
            
            # Format duration as MM:SS
            mins = duration // 60
            secs = duration % 60
            duration_text = f"{mins:02d}:{secs:02d}"
            
            w2, h2 = self._measure_text(draw, duration_text, self.font_medium)
            x2 = (128 - w2) // 2
            y2 = (64 - h2) // 2 + 12
            draw.text((x2, y2), duration_text, font=self.font_medium, fill=255)
        else:
            # Single line: rhythm only
            w, h = self._measure_text(draw, display_text, self.font_large)
            x = (128 - w) // 2
            y = (64 - h) // 2
            draw.text((x, y), display_text, font=self.font_large, fill=255)
        
        self.device.display(img)
    
    def show_idle(self):
        """Show idle state - ready to record with button menu"""
        self.current_state = "IDLE"
        if not self.device:
            return

        # Create image for button menu
        img = Image.new("1", (128, 64), color=0)
        draw = ImageDraw.Draw(img)
        
        # Title
        title = "MENU"
        w, h = self._measure_text(draw, title, self.font_medium)
        x = (128 - w) // 2
        draw.text((x, 2), title, font=self.font_medium, fill=255)
        
        # Button options (using small font for 3 lines)
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
        """Show recording in progress"""
        self.current_state = "RECORDING"
        self.show_message("RECORDING", size="medium")
    
    def show_countdown(self, count: int):
        """Show countdown before recording starts"""
        self.current_state = "COUNTDOWN"
        self.show_message("RECORDING IN", str(count), size="medium")
    
    def show_processing(self):
        """Show processing state"""
        self.current_state = "PROCESSING"
        self.show_message("ANALYZING", "ECG DATA", size="medium")
    
    def show_stopping(self, rhythm: str = None, duration: int = 0):
        """Show 'STOPPING...' overlaid on the current rhythm display"""
        self.current_state = "STOPPING"
        if not self.device:
            return
        
        img = Image.new("1", (128, 64), color=0)
        draw = ImageDraw.Draw(img)
        
        if rhythm and duration > 0:
            # Show rhythm + duration like normal, but add STOPPING... banner
            rhythm_map = {
                'NSR': 'NORMAL', 'AFIB': 'A-FIB', 'PVC': 'PVC',
                'LBBB': 'LBBB', 'NORMAL': 'NORMAL'
            }
            display_text = rhythm_map.get(rhythm, rhythm)
            
            w, h = self._measure_text(draw, display_text, self.font_large)
            x = (128 - w) // 2
            draw.text((x, 2), display_text, font=self.font_large, fill=255)
            
            mins, secs = duration // 60, duration % 60
            duration_text = f"{mins:02d}:{secs:02d}"
            w2, h2 = self._measure_text(draw, duration_text, self.font_small)
            x2 = (128 - w2) // 2
            draw.text((x2, 32), duration_text, font=self.font_small, fill=255)
            
            # STOPPING... banner at bottom
            stop_text = "STOPPING..."
            ws, hs = self._measure_text(draw, stop_text, self.font_small)
            xs = (128 - ws) // 2
            draw.text((xs, 50), stop_text, font=self.font_small, fill=255)
        else:
            # No rhythm context — just show STOPPING...
            self.show_message("STOPPING...", size="medium")
            return
        
        self.device.display(img)
    
    def show_error(self, error_msg: str):
        """Show error message"""
        self.current_state = "ERROR"
        self.show_message("ERROR", error_msg, size="small")
    
    def show_initializing(self):
        """Show boot-time initializing screen (displayed until all processes ready)"""
        self.current_state = "INITIALIZING"
        if not self.device:
            return

        img = Image.new("1", (128, 64), color=0)
        draw = ImageDraw.Draw(img)

        title = "ECG SYSTEM"
        wt, ht = self._measure_text(draw, title, self.font_medium)
        draw.text(((128 - wt) // 2, 8), title, font=self.font_medium, fill=255)

        sub = "Initializing..."
        ws, hs = self._measure_text(draw, sub, self.font_small)
        draw.text(((128 - ws) // 2, 36), sub, font=self.font_small, fill=255)

        self.device.display(img)

    def update_loop(self, status_queue: mp.Queue, control_queue: mp.Queue):
        """
        Main update loop - listens for status messages and updates display.
        
        Args:
            status_queue: Input queue for status messages from pipeline
            control_queue: Input queue for control commands
        """
        self.running = True
        self.show_initializing()
        
        while self.running:
            try:
                # Check for status messages
                msg_type, msg_content = status_queue.get(timeout=0.1)
                
                if msg_type == 'STATUS':
                    if msg_content == 'INIT_READY':
                        self.show_idle()
                    elif msg_content.startswith('COUNTDOWN:'):
                        count = int(msg_content.split(':')[1])
                        self.show_countdown(count)
                    elif msg_content == 'RECORDING_START':
                        pass  # Skip redundant recording screen, countdown already showed intent
                    elif msg_content == 'RECORDING_STOP':
                        # Show STOPPING... overlaid on current rhythm display
                        self.show_stopping(self.current_rhythm, self.current_duration)
                    elif msg_content.startswith('BUFFERING'):
                        if self.current_state == 'STOPPING':
                            pass  # Don't overwrite STOPPING... display
                        elif ':' in msg_content:
                            dur = int(msg_content.split(':')[1])
                            self.current_duration = dur
                            mins, secs = dur // 60, dur % 60
                            self.show_message("BUFFERING", f"ECG {mins:02d}:{secs:02d}", size="medium")
                        else:
                            self.show_message("BUFFERING", "ECG...", size="medium")
                    elif msg_content.startswith('SAVED:'):
                        # Extract filename and duration
                        parts = msg_content.split(':', 2)  # Split into max 3 parts
                        if len(parts) >= 3:
                            full_filename = parts[1].strip()
                            duration = int(parts[2].strip())
                            
                            # Format: ecg_DDMMYY_HHMMSS -> show DDMMYY-HHMMSS
                            if full_filename.startswith('ecg_'):
                                compact = full_filename[4:].replace('_', '-')
                            else:
                                compact = full_filename[-15:]
                            
                            # Format duration as MM:SS
                            mins = duration // 60
                            secs = duration % 60
                            duration_text = f"{mins:02d}:{secs:02d}"
                            
                            # Show filename on line 1, duration on line 2
                            self.show_message("REC", "STOPPED", size="medium")
                            time.sleep(1.5)
                            self.show_message(compact, duration_text, size="small")
                        else:
                            # Fallback without duration
                            full_filename = parts[1].strip() if len(parts) > 1 else "unknown"
                            if full_filename.startswith('ecg_'):
                                compact = full_filename[4:].replace('_', '-')
                            else:
                                compact = full_filename[-15:]
                            self.show_message("REC", "STOPPED", size="medium")
                            time.sleep(1.5)
                            self.show_message("SAVED", compact, size="small")

                        time.sleep(3)  # Display save name for 3 seconds
                        self.show_idle()  # Return to menu
                    elif msg_content.startswith('RHYTHM:'):
                        # Extract rhythm classification and optional duration
                        parts = msg_content.split(':', 2)
                        rhythm = parts[1].strip()
                        duration = int(parts[2].strip()) if len(parts) > 2 else 0
                        self.current_duration = duration
                        if self.current_state == 'STOPPING':
                            pass  # Don't overwrite STOPPING... display
                        else:
                            self.show_rhythm(rhythm, duration)
                    elif msg_content == 'SHUTTING_DOWN':
                        # Show shutdown message
                        self.show_message("SHUTTING", "DOWN...", size="medium")
                    elif msg_content == 'USB_TRANSFER_START':
                        # Show USB transfer starting
                        self.show_message("USB", "TRANSFER...", size="medium")
                    elif msg_content.startswith('USB_TRANSFER_COMPLETE'):
                        # Extract file count if provided
                        parts = msg_content.split(':', 1)
                        if len(parts) > 1:
                            count = parts[1].strip()
                            self.show_message("TRANSFER OK", f"{count} files", size="medium")
                        else:
                            self.show_message("TRANSFER", "COMPLETE", size="medium")
                        time.sleep(3)  # Show for 3 seconds
                        self.show_idle()
                    elif msg_content == 'USB_TRANSFER_FAILED':
                        # Show transfer failure
                        self.show_message("TRANSFER", "FAILED", size="medium")
                        time.sleep(3)
                        self.show_idle()
                    elif msg_content == 'CLEAR_DISPLAY':
                        # Clear display (for shutdown)
                        if self.device:
                            self.device.clear()
                            self.device.show()
                
                elif msg_type == 'ERROR':
                    self.show_error(msg_content[:20])  # Truncate long errors
                
            except mp.queues.Empty:
                continue
            except KeyboardInterrupt:
                break
            except Exception as e:
                # I2C bus error (e.g. buzzer interference knocked OLED off bus)
                # Attempt re-initialization rather than crashing the whole process
                print(f"[OLED] I2C error during display update: {e}")
                print(f"[OLED] Attempting re-initialization...")
                self.device = None
                for attempt in range(5):
                    time.sleep(1.0)
                    try:
                        self.setup()
                        print(f"[OLED] Re-initialized successfully (attempt {attempt + 1})")
                        break
                    except Exception as reinit_err:
                        print(f"[OLED] Re-init attempt {attempt + 1} failed: {reinit_err}")
                else:
                    print("[OLED] Re-initialization failed after 5 attempts, display disabled")
                    # Keep running without display so pipeline continues
                    self.device = None
                    # Swallow all future display errors silently
                    continue
    
    def cleanup(self):
        """Clean up display"""
        if self.device:
            try:
                self.device.clear()
                self.device.show()
            except:
                pass


def oled_worker(status_queue: mp.Queue, control_queue: mp.Queue):
    """
    Worker function for OLED display process.
    
    Args:
        status_queue: Input queue for status messages from pipeline
        control_queue: Input queue for control commands
    """
    display = OLEDDisplay()
    
    try:
        display.setup()
        display.update_loop(status_queue, control_queue)
    except Exception as e:
        import traceback
        print(f"[OLED] Error: {e}")
        print(f"[OLED] Traceback:")
        traceback.print_exc()
    finally:
        display.cleanup()
