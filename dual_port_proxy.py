"""
Dual Port Proxy for Streamlit Deployments (Dokploy / Coolify / Traefik / Railway).
Guarantees that both port 8501 (Streamlit standard) and port 3000 (Dokploy UI default)
are active and respond with zero-configuration.
"""
import os
import sys
import socket
import threading
import subprocess

def pipe(source, target):
    try:
        while True:
            data = source.recv(8192)
            if not data:
                break
            target.sendall(data)
    except Exception:
        pass
    finally:
        try:
            source.close()
        except Exception:
            pass
        try:
            target.close()
        except Exception:
            pass

def forward_connection(client_conn, target_port):
    try:
        target_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        target_conn.connect(('127.0.0.1', target_port))
        threading.Thread(target=pipe, args=(client_conn, target_conn), daemon=True).start()
        threading.Thread(target=pipe, args=(target_conn, client_conn), daemon=True).start()
    except Exception:
        try:
            client_conn.close()
        except Exception:
            pass

def start_secondary_listener(proxy_port, target_port):
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', proxy_port))
        server.listen(128)
        while True:
            client_conn, _ = server.accept()
            threading.Thread(target=forward_connection, args=(client_conn, target_port), daemon=True).start()
    except Exception:
        pass

def main():
    raw_port = os.environ.get("PORT", "").strip()
    try:
        target_port = int(raw_port) if raw_port else 8501
    except ValueError:
        target_port = 8501

    # Bridge all standard PaaS ports (8501 Streamlit, 3000 Dokploy, 80 HTTP, 8080 Cloud)
    # to target_port so any Dokploy port setting connects instantly
    common_ports = [8501, 3000, 80, 8080]
    for p in common_ports:
        if p != target_port:
            threading.Thread(target=start_secondary_listener, args=(p, target_port), daemon=True).start()

    cmd = [
        sys.executable, "-m", "streamlit", "run", "xgboost_trader.py",
        f"--server.port={target_port}",
        "--server.address=0.0.0.0"
    ]
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
