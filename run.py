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

import config          # noqa: F401  - imported for its Python version check

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


def _cert_via_python(key, crt, ip):
    """Generate a self-signed certificate using `cryptography`.

    Preferred over shelling out to openssl, which is not installed by default
    on Windows.
    """
    from datetime import datetime, timedelta, timezone
    import ipaddress
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "potholesense")])
    alt = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    try:
        alt.append(x509.IPAddress(ipaddress.ip_address(ip)))
    except ValueError:
        pass
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(k.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(alt), critical=False)
            .sign(k, hashes.SHA256()))

    key.write_bytes(k.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _cert_via_openssl(key, crt, ip):
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key), "-out", str(crt), "-days", "365",
        "-subj", "/CN=potholesense",
        "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost",
    ], check=True)


def ensure_cert():
    """Create a self-signed certificate if we do not already have one.

    HTTPS is not optional: browsers refuse camera and geolocation access on
    plain HTTP from anything other than localhost, and the phone reaches this
    server over the LAN.
    """
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    key, crt = CERT_DIR / "key.pem", CERT_DIR / "cert.pem"
    if key.exists() and crt.exists():
        return key, crt

    ip = lan_ip()
    try:
        _cert_via_python(key, crt, ip)
    except ImportError:
        try:
            _cert_via_openssl(key, crt, ip)
        except (OSError, subprocess.CalledProcessError):
            print("ERROR: could not generate a certificate.\n"
                  "       Install the cryptography package:\n"
                  "           pip install cryptography\n"
                  "       (or run without --https, but the phone camera "
                  "will not work)", file=sys.stderr)
            raise SystemExit(1)
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
