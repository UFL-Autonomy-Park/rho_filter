import http.server
import socketserver
import mimetypes
import os

# Force correct Wasm MIME type
mimetypes.init()
mimetypes.add_type('application/wasm', '.wasm')

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="test/gui_test", **kwargs)

print("Serving on http://localhost:8000")
with socketserver.TCPServer(("", 8000), Handler) as httpd:
    httpd.serve_forever()
