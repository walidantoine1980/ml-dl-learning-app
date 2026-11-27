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
    """Transfère les octets de source vers target en préservant le half-close TCP (SHUT_WR)."""
    try:
        while True:
            data = source.recv(65536)
            if not data:
                break
            target.sendall(data)
    except Exception:
        pass
    finally:
        # Half-close uniquement le sens écriture vers target : ne ferme pas le socket retour !
        try:
            target.shutdown(socket.SHUT_WR)
        except Exception:
            pass

def forward_connection(client_conn, target_port):
    target_conn = None
    try:
        client_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass
    try:
        target_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        target_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        target_conn.connect(('127.0.0.1', target_port))
        t1 = threading.Thread(target=pipe, args=(client_conn, target_conn), daemon=True)
        t2 = threading.Thread(target=pipe, args=(target_conn, client_conn), daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    except Exception:
        pass
    finally:
        for conn in (client_conn, target_conn):
            if conn is not None:
                try:
                    conn.close()
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
