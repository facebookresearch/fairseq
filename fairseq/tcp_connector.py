# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import atexit
import pickle
import socket
import socketserver
import threading
import time
import faulthandler

class TcpConnector(object):
    """Synchronize data across multiple nodes over TCP."""

    SOCKET_TIMEOUT=300
    MESSAGE_QUEUE_SIZE = 10

    def __init__(self, port, rank, world_size, master_host):
        self.port = port
        self.root = master_host
        self.nhosts = world_size
        self.host_idx = rank
        self.server = None
        self.current_message_id = 0
        self.socket = None
        faulthandler.enable(all_threads=True)
        if rank == 0:
            self._create_server()

    def _create_server(self):
        messages = {}
        condition = threading.Condition()
        nhosts = self.nhosts
        class TCPHandler(socketserver.BaseRequestHandler):
            def handle(self):
                # Keep socket open forever
                while True:
                    rcvd = TcpConnector.recv_msg(self.request)
                    if rcvd is None:
                        return
                    id, host_idx, data = rcvd
                    with condition:
                        if not id in messages:
                            messages[id] = [None] * nhosts
                            k = id - TcpConnector.MESSAGE_QUEUE_SIZE
                            if k in messages:
                                del messages[k]

                        messages[id][host_idx] = data
                        condition.wait_for(lambda : sum(1 for x in messages[id] if x is None) == 0,
                                           timeout=TcpConnector.SOCKET_TIMEOUT)
                        condition.notify_all()

                    TcpConnector.send_msg(self.request, messages[id])


        # HOST='' means running on interface 0.0.0.0 that seems to work for everyone
        self.server = socketserver.ThreadingTCPServer(('', self.port), TCPHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        print("Server is running on {}:{}".format(socket.gethostname(), self.port), flush=True)

        @atexit.register
        def _cleanup():
            self.server.shutdown()
            self.server.server_close()
            self.server_thread.join()

    def all_gather(self, message):
        """Gathers messages from all nodes into a list."""
        for retry in range(8):
            if retry > 0:
                print("Retry {}, message {}".format(retry, message), flush=True)
                time.sleep(2 ** retry)
            try:
                if self.socket is None:
                    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.socket.settimeout(TcpConnector.SOCKET_TIMEOUT)
                    self.socket.connect((self.root, self.port))
                TcpConnector.send_msg(self.socket, (self.current_message_id, self.host_idx, message))
                received = TcpConnector.recv_msg(self.socket)
                self.current_message_id += 1
                return received
            except socket.timeout:
                print("Socket timeout", flush=True)
                TcpConnector.close(self.socket)
            except ConnectionError:
                print("Unable to connect to {}:{}, message_id {}, host_idx {}, message {}".format(
                    self.root, self.port, self.current_message_id, self.host_idx, message), flush=True)
                TcpConnector.close(self.socket)
            except Exception as e:
                print("Unexpected exception {}".format(e))
                break

        raise Exception("Unable send the message to the root node")

    @staticmethod
    def close(socket):
        if socket:
            try:
                socket.close()
            except:
                print("Unable to close socket")


    @staticmethod
    def send_msg(stream, message):
        enc = pickle.dumps(message)
        stream.sendall(len(enc).to_bytes(8, byteorder='big'))
        stream.sendall(enc)

    @staticmethod
    def recv_msg(stream):
        size = int.from_bytes(stream.recv(8), byteorder='big')
        if size == 0:
            print('Shutdown request received', flush=True)
            return None
        enc = stream.recv(size)
        while len(enc) < size:
            enc += stream.recv(size - len(enc))
        data = pickle.loads(enc)
        return data

    def shutdown(self):
        if self.socket:
            TcpConnector.close(self.socket)
        if self.host_idx == 0:
            self.server.shutdown()
