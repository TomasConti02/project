import socket

HOST = '0.0.0.0'  # ascolta su tutte le interfacce
PORT = 12345      # porta su cui il server ascolta

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"Server in ascolto su {HOST}:{PORT}")

    while True:
        conn, addr = server_socket.accept()
        with conn:
            print(f"Connessione ricevuta da {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                print(f"Ricevuto: {data.decode()}")
                conn.sendall(data)
