import os
import threading
import webbrowser

import uvicorn

HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8765"))

if __name__ == "__main__":
    if os.environ.get("NO_BROWSER") != "1":
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    uvicorn.run("app.main:app", host=HOST, port=PORT, log_level="info")
