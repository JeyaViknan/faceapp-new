#!/usr/bin/env python3
"""
Startup script for Render deployment
"""
import os
import uvicorn
from app1_simple import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app1_simple:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info"
    )
