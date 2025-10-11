import socket
import os
import time
import threading
import tempfile
import shutil
import uuid
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
import yaml
import logging
import cgi
from io import BytesIO
from utils.xmp2lua import parse_xmp

# Load configuration
def load_config():
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

class LightroomAPI:
    def __init__(self, host=None, port=None):
        self.host = host or config['socket']['host']
        self.port = port or config['socket']['port']
        self.socket = None
        self.response_thread = None
        self.connected = False
    
    def connect(self):
        """Establish connection to server"""
        if self.connected:
            return True
            
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(config['socket']['timeout'])
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            
            keepalive = config['socket']['keepalive']
            if hasattr(socket, 'TCP_KEEPIDLE'):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, keepalive['idle'])
            if hasattr(socket, 'TCP_KEEPINTVL'):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, keepalive['interval'])
            if hasattr(socket, 'TCP_KEEPCNT'):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, keepalive['count'])
            
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            # Start response handling thread
            self.response_thread = threading.Thread(target=self._handle_responses)
            self.response_thread.daemon = True
            self.response_thread.start()
            
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            self.connected = False
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
            return False
    
    def _handle_responses(self):
        """Handle server responses"""
        while self.connected:
            try:
                response = self.socket.recv(config['connection']['buffer_size']).decode().strip()
                if not response:
                    print("Empty response from server")
                    self._disconnect()
                    break
                
                self._process_response(response)
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error in response handler: {e}")
                self._disconnect()
                break
    
    def _process_response(self, response):
        """Process server response messages"""
        print(f"Received response: {response}")
        
        # Handle simple responses
        if response == "ok":
            print("Request acknowledged by Lightroom")
            return
            
        # Handle structured responses
        if '|' in response:
            status, *message = response.split('|')
            message = '|'.join(message) if message else ''
            
            if status == "success":
                print("\nPhoto processed successfully!")
                if message:
                    print(f"Output saved to: {message}")
            elif status == "error":
                print(f"\nError: {message}")
            elif status == "pong":
                print("Server is alive (received pong)")
            elif status == "processing":
                print(f"Processing started: {message}")
        else:
            print(f"Unknown response format: {response}")
    
    def _disconnect(self):
        """Internal method to handle disconnection"""
        self.connected = False
    
    def send_request(self, message):
        """Send request to server"""
        if not self.connected and not self.connect():
            raise ConnectionError("Failed to connect to server")
            
        try:
            self.socket.sendall((message + "\n").encode())
            return True
        except Exception as e:
            print(f"Error sending request: {e}")
            self._disconnect()
            raise
    
    def process_photo(self, photo_path, lua_path):
        """Process photo with lua preset"""
        # Verify files exist
        if not os.path.exists(photo_path):
            raise FileNotFoundError(f"Photo file not found: {photo_path}")
        if not os.path.exists(lua_path):
            raise FileNotFoundError(f"Lua preset file not found: {lua_path}")
            
        # Use pipe-separated format
        message = f"process|{Path(photo_path).resolve()}|{Path(lua_path).resolve()}"
        print(f"Sending to Lightroom: {message}")
        
        return self.send_request(message)
    
    def close(self):
        """Close connection"""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            finally:
                self.socket = None

def test_server_connection(api):
    """Test server connection"""
    if not api.connect():
        return False
        
    # Send ping request
    result = api.send_request("ping")
    if result:
        print("Server is running")
        return True
    
    print("Could not connect to server")
    return False

class PhotoProcessHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_type = self.headers.get('Content-Type', '')
            
            if content_type.startswith('multipart/form-data'):
                self._handle_file_upload()
            elif content_type == 'application/json':
                self._handle_json_request()
            else:
                self._send_error_response(400, "Unsupported content type")
                
        except Exception as e:
            self._send_error_response(500, str(e))
    
    def _handle_json_request(self):
        """Handle traditional JSON requests with file paths"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            photo_path = data.get('photo_path')
            preset_path = data.get('lua_path')  # Can be .lua or .xmp

            if not photo_path or not preset_path:
                self._send_error_response(400, "Missing photo_path or lua_path")
                return

            # Check if preset is XMP and convert to Lua
            if preset_path.lower().endswith('.xmp'):
                print(f"Converting XMP to Lua: {preset_path}")
                try:
                    lua_table = parse_xmp(preset_path)
                    lua_path = preset_path.replace('.xmp', '.lua')
                    with open(lua_path, 'w', encoding='utf-8') as f:
                        f.write('return ' + lua_table)
                    print(f"Converted XMP to Lua: {lua_path}")
                    preset_path = lua_path
                except Exception as e:
                    self._send_error_response(500, f"XMP conversion failed: {str(e)}")
                    return

            print(f"Processing in local mode: {photo_path}, {preset_path}")
            self.server.lightroom_api.process_photo(photo_path, preset_path)
            self._send_success_response()

        except json.JSONDecodeError:
            self._send_error_response(400, "Invalid JSON")
        except FileNotFoundError as e:
            self._send_error_response(404, str(e))
        except ConnectionError as e:
            self._send_error_response(503, str(e))
    
    def _handle_file_upload(self):
        """Handle multipart file upload requests"""
        temp_files = []
        try:
            # Check content length against max total size
            content_length = int(self.headers.get('Content-Length', 0))
            max_total_size = config.get('upload', {}).get('max_total_size', 209715200)
            
            if content_length > max_total_size:
                self._send_error_response(413, f"Request too large. Max size: {max_total_size:,} bytes")
                return
            
            # Parse multipart form data
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            
            # Check if this is upload mode
            mode = form.getvalue('mode')
            if mode != 'upload':
                self._send_error_response(400, "Invalid mode")
                return
            
            # Get uploaded files
            photo_file = form['photo_file']
            preset_file = form['lua_file']  # Can be .lua or .xmp
            photo_filename = form.getvalue('photo_filename')
            preset_filename = form.getvalue('lua_filename')  # Can be .lua or .xmp

            if not photo_file.file or not preset_file.file:
                self._send_error_response(400, "Missing files")
                return

            # Validate file extensions
            upload_config = config.get('upload', {})
            allowed_photo_ext = upload_config.get('allowed_photo_extensions', [])
            allowed_preset_ext = upload_config.get('allowed_preset_extensions', [])
            max_file_size = upload_config.get('max_file_size', 104857600)

            photo_ext = os.path.splitext(photo_filename)[1].lower()
            preset_ext = os.path.splitext(preset_filename)[1].lower()

            if photo_ext not in allowed_photo_ext:
                # print(allowed_photo_ext)
                # print(photo_ext)
                self._send_error_response(400, f"Invalid photo file type: {photo_ext}")
                return

            if preset_ext not in allowed_preset_ext:
                self._send_error_response(400, f"Invalid preset file type: {preset_ext}")
                return

            print(f"Processing in remote mode: {photo_filename}, {preset_filename}")
            
            # Create upload directory
            upload_dir = upload_config.get('upload_dir', 'uploads')
            dir_prefix = upload_config.get('dir_prefix', 'upload_')
            
            # Create base upload directory if it doesn't exist
            project_root = Path(__file__).parent
            base_upload_dir = project_root / upload_dir
            base_upload_dir.mkdir(exist_ok=True)
            
            # Create unique subdirectory for this upload
            unique_id = str(uuid.uuid4())[:8]
            upload_subdir = base_upload_dir / f"{dir_prefix}{unique_id}"
            upload_subdir.mkdir(exist_ok=True)
            
            print(f"Upload directory: {upload_subdir}")
            
            # Save uploaded files to upload directory
            photo_temp_path = upload_subdir / photo_filename
            preset_temp_path = upload_subdir / preset_filename

            # Write photo file with size check
            with open(photo_temp_path, 'wb') as f:
                bytes_written = 0
                while True:
                    chunk = photo_file.file.read(8192)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > max_file_size:
                        raise ValueError(f"Photo file too large. Max size: {max_file_size:,} bytes")
                    f.write(chunk)
            temp_files.append(photo_temp_path)

            # Write preset file with size check
            with open(preset_temp_path, 'wb') as f:
                bytes_written = 0
                while True:
                    chunk = preset_file.file.read(8192)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > max_file_size:
                        raise ValueError(f"Preset file too large. Max size: {max_file_size:,} bytes")
                    f.write(chunk)
            temp_files.append(preset_temp_path)

            print(f"Uploaded files saved to directory: {upload_subdir}")
            print(f"Photo: {photo_temp_path} ({os.path.getsize(str(photo_temp_path)):,} bytes)")
            print(f"Preset: {preset_temp_path} ({os.path.getsize(str(preset_temp_path)):,} bytes)")

            # Check if preset is XMP and convert to Lua
            if preset_ext == '.xmp':
                print(f"Converting uploaded XMP to Lua: {preset_temp_path}")
                try:
                    lua_table = parse_xmp(str(preset_temp_path))
                    lua_temp_path = upload_subdir / preset_filename.replace('.xmp', '.lua')
                    with open(lua_temp_path, 'w', encoding='utf-8') as f:
                        f.write('return ' + lua_table)
                    temp_files.append(lua_temp_path)
                    print(f"Converted XMP to Lua: {lua_temp_path}")
                    preset_temp_path = lua_temp_path
                except Exception as e:
                    self._send_error_response(500, f"XMP conversion failed: {str(e)}")
                    return

            # Process the files (same as local mode)
            # Convert to strings and normalize paths for Lightroom
            photo_path_str = str(photo_temp_path).replace('\\', '/')
            preset_path_str = str(preset_temp_path).replace('\\', '/')
            print(f"Processing in remote mode: {photo_path_str}, {preset_path_str}")
            self.server.lightroom_api.process_photo(photo_path_str, preset_path_str)
            self._send_success_response()
            
        except KeyError as e:
            self._send_error_response(400, f"Missing form field: {str(e)}")
        except Exception as e:
            self._send_error_response(500, f"Upload processing error: {str(e)}")
        finally:
            # Handle file cleanup based on config
            keep_files = upload_config.get('keep_files', True)
            if keep_files:
                print(f"Upload files preserved:")
                for temp_file in temp_files:
                    if os.path.exists(str(temp_file)):
                        print(f"  - {temp_file}")
            else:
                print(f"Cleaning up upload files...")
                self._cleanup_temp_files(temp_files)
    
    def _send_success_response(self):
        """Send successful response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "success"}).encode())
    
    def _send_error_response(self, code, message):
        """Send error response"""
        self.send_error(code, message)
    
    def _cleanup_temp_files(self, temp_files):
        """Clean up uploaded files and directories"""
        for temp_file in temp_files:
            try:
                temp_file_str = str(temp_file)
                if os.path.exists(temp_file_str):
                    os.remove(temp_file_str)
                    print(f"Cleaned up file: {temp_file}")
                    
                # Also remove the upload directory if empty
                upload_dir = os.path.dirname(temp_file_str)
                if os.path.exists(upload_dir) and not os.listdir(upload_dir):
                    os.rmdir(upload_dir)
                    print(f"Cleaned up directory: {upload_dir}")
            except Exception as e:
                print(f"Warning: Failed to cleanup {temp_file}: {e}")

def run_http_server(api, port=None):
    """Run HTTP server"""
    port = port or config['http']['port']
    server = HTTPServer((config['http']['host'], port), PhotoProcessHandler)
    server.lightroom_api = api
    print(f"Starting HTTP server on port {port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nHTTP server stopped")
    finally:
        server.server_close()

def main():
    # Create API client
    api = LightroomAPI()
    
    try:
        # Test connection
        if not test_server_connection(api):
            print("Error: Could not connect to the Lightroom plugin server")
            return
            
        # Start HTTP server
        http_thread = threading.Thread(target=run_http_server, args=(api,))
        http_thread.daemon = True
        http_thread.start()
        
        print("\nHTTP server is running. You can now send POST requests to process photos.")
        print("Example POST request to http://localhost:7777:")
        print('''
        {
            "photo_path": "path/to/photo.dng",
            "lua_path": "path/to/preset.lua"
        }
        ''')
        
        # Keep main program running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        api.close()

if __name__ == "__main__":
    main() 