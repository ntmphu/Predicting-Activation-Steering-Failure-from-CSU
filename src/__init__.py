import sys

# Windows terminals default to a legacy encoding that can't print symbols like Δ or v̄
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")
