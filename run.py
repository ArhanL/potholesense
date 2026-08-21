#!/usr/bin/env python3
"""Start the PotholeSense server.

Usage:
    python run.py                 # http on all interfaces, port 8000
    python run.py --https         # self-signed TLS (needed for phone GPS/camera)
"""
import argparse
import socket
import subprocess
import sys
from pathlib import Path

CERT_DIR = Path(__file__).parent / "data" / "certs"


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert():
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    key, crt = CERT_DIR / "key.pem", CERT_DIR / "cert.pem"
    if key.exists() and crt.exists():
        return key, crt
    ip = lan_ip()
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key), "-out", str(crt), "-days", "365",
        "-subj", "/CN=potholesense",
        "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost",
    ], check=True)
    print(f"Generated self-signed certificate for {ip}")
    return key, crt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--https", action="store_true",
                    help="serve over TLS - required for camera/GPS on a phone")
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    import uvicorn
    kwargs = dict(host=args.host, port=args.port, reload=args.reload)
    scheme = "http"
    if args.https:
        key, crt = ensure_cert()
        kwargs.update(ssl_keyfile=str(key), ssl_certfile=str(crt))
        scheme = "https"

    ip = lan_ip()
    print("\n" + "=" * 58)
    print("  PotholeSense")
    print(f"  Phone (capture):  {scheme}://{ip}:{args.port}/")
    print(f"  Laptop (map):     {scheme}://{ip}:{args.port}/dashboard")
    print("=" * 58 + "\n")
    uvicorn.run("app.main:app", **kwargs)


if __name__ == "__main__":
    sys.exit(main())
