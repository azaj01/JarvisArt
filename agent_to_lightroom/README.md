# Agent-to-Lightroom Protocol

A Python API client for communicating with Adobe Lightroom via a custom plugin, enabling automated photo processing through HTTP requests. Supports both local (file path) and remote (file upload) modes.

## Components

- **XMPlayer.lrplugin/**: Lightroom plugin that creates a socket server
- **lightroom_api_server.py**: Python server that connects to the plugin and provides HTTP API
- **test_http.py**: Client for sending photo processing requests (supports both local and remote modes)
- **config/**: Configuration files in YAML format

## Quick Start

1. Install the plugin to your Lightroom :

   - open Lightroom Classic, click `File`-`Plug-in Manager`-`Add`

   - choose   `XMPlayer.lrplugin/`  directory

2. Run the Python API server:
   ```bash
   python lightroom_api_server.py
   ```

3. Configure paths in `config/paths.yaml` according to your setup

4. Send photo processing requests:

   ```bash
   python test_http.py
   ```

## Configuration

The project uses YAML configuration files located in the `config/` directory:

### config/config.yaml
- Socket connection settings (host, port, timeouts)
- HTTP server settings
- Keepalive and buffer configurations
- Logging settings
- **File upload limits and allowed extensions**

### config/paths.yaml
- Example photo and preset paths
- Default directories
- Supported file formats

### Upload Configuration
- Max file size: 100MB per file
- Max total upload: 200MB
- Supported photo formats: .jpg, .jpeg, .dng, .cr2, .nef, .tiff, .tif, .raw, .arw
- Supported preset formats: .lua, .xmp
