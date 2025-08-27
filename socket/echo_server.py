import socket

HOST = '127.0.0.1'
PORT = 65432

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Worker in ascolto su {HOST}:{PORT}")

    while True:
        conn, addr = s.accept()
        with conn:
            data = conn.recv(1024)
            if not data:
                continue
            text = data.decode()
            print(f"Ricevuto dal client: {text}")

            # Trasforma in maiuscolo
            uppercase_text = text.upper()
            conn.sendall(uppercase_text.encode())
