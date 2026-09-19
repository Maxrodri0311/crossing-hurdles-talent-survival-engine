"""
src/web_dashboard.py - Interactive Policy Simulation Web Server Launcher
========================================================================
Lightweight entry point to serve and open the Crossing Hurdles Interactive
Web Simulator (web/index.html) locally via Python standard library http.server.
"""

import os
import sys
import webbrowser
import http.server
import socketserver
import threading
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
WEB_DIR = os.path.join(PROJECT_ROOT, "web")


class CustomHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving web/ directory with CORS and clean MIME types."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format, *args):
        # Suppress noisy standard request logging in console
        pass


def launch_web_server(port: int = 8080, open_browser: bool = True, timeout_sec: int = 0):
    """
    Launches a local static server serving the Interactive Simulator.
    If timeout_sec > 0, server shuts down after timeout (useful for automated testing/CLI).
    """
    os.chdir(PROJECT_ROOT)
    html_file = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(html_file):
        raise FileNotFoundError(f"Web interface not found at {html_file}")

    url = f"http://localhost:{port}/index.html"
    print("=" * 80)
    print("⚡ CROSSING HURDLES: INTERACTIVE DECISION & CONFORMAL SIMULATOR")
    print(f"🌐 Server running at: {url}")
    print(f"📁 Serving directory: {WEB_DIR}")
    print("=" * 80)

    # Allow port reuse to avoid Address Already In Use
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), CustomHTTPHandler) as httpd:
        if open_browser:
            threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url)), daemon=True).start()
        
        if timeout_sec > 0:
            timer = threading.Timer(timeout_sec, httpd.shutdown)
            timer.start()
            print(f"[!] Server will auto-terminate in {timeout_sec}s...")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[!] Shutting down web server.")
        finally:
            httpd.server_close()


if __name__ == "__main__":
    port_arg = 8080
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port_arg = int(sys.argv[1])
    
    # Check if run with --no-browser
    auto_open = "--no-browser" not in sys.argv
    timeout = 0
    for arg in sys.argv:
        if arg.startswith("--timeout="):
            timeout = int(arg.split("=")[1])

    launch_web_server(port=port_arg, open_browser=auto_open, timeout_sec=timeout)
