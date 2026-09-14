import socket
import struct
import threading
from enum import Enum

from PySide6.QtCore import QObject, Signal
 
class Types(Enum):
    TbBool = 7
    TbInt64 = 8
    TbInt32 = 9
    TbFloat = 10
    TbDouble = 11
    TbString = 12
    TbChar = 13
    TbTB = 14
    TbBinary = 15

class MsgType(Enum):
    # Version 2 has separate message numbers: old plugins cannot decode pairs
    # as filenames or accidentally route instance commands to legacy objects.
    MSG_PROTOCOL_VERSION = 4711
    MSG_LOAD_NEURONS = 4720
    MSG_SHOW_NEURONS = 4721
    MSG_HIDE_NEURONS = 4722
    MSG_COLOR_NEURONS = 4723
    MSG_LOAD_VOLUME = 4704
    MSG_SHOW_VOLUME = 4705
    MSG_HIDE_VOLUME = 4706
    MSG_VIEW_ALL = 4707
    MSG_TEST = 4708
    MSG_SELECT_NEURONS = 4724

class MessageClient(QObject):

    log_signal = Signal(str)
    selection_signal = Signal(list)  # list of v2 instance IDs
    disconnected_signal = Signal()


    def __init__(self):
        super().__init__()
        self.sock = None
        self.connected = False
        self.recv_thread = None

    def connect(self, host='134.95.85.134', port=33333): #host='134.95.85.134'
        try:
            self.sock = socket.create_connection((host, port), timeout=5.0)
            self.log_signal.emit("Connected to server")

            # Step 1: Wait for byte order handshake byte (\x01)
            handshake = self.sock.recv(1)
            if handshake != b'\x01':
                self.log_signal.emit("Unexpected handshake byte from server")
                raise ValueError("Unexpected byte order handshake")
            self.log_signal.emit("Received handshake byte")

            # Step 2: Echo it back
            self.sock.sendall(handshake)
            self.log_signal.emit("Sent handshake byte back")

            self._negotiate_instance_protocol()
            self.sock.settimeout(None)
            self.connected = True
            self.recv_thread = threading.Thread(
                target=self.receive_loop, args=(self.sock,), daemon=True
            )
            self.recv_thread.start()
            return True

        except Exception as e:
            if self.sock is not None:
                self.sock.close()
                self.sock = None
            self.connected = False
            self.log_signal.emit(f"Connection failed: {e}")
            return False

    def _negotiate_instance_protocol(self):
        """Require the companion plugin before allowing any neuron messages."""
        payload = b'\x01' + struct.pack('<Bi', Types.TbInt32.value, 2)
        header = struct.pack('>iiii', 8564, 0, MsgType.MSG_PROTOCOL_VERSION.value, len(payload))
        self.sock.sendall(header + payload)
        try:
            header = self._recv_exact(self.sock, 16)
            if header is None:
                raise ValueError("server closed the connection")
            _, _, message_type, size = struct.unpack('>iiii', header)
            if message_type != MsgType.MSG_PROTOCOL_VERSION.value or size not in (5, 6):
                raise ValueError("unexpected protocol response")
            response = self._recv_exact(self.sock, size)
            if response is not None and size == 6 and response[0] in (0, 1):
                response = response[1:]
            if response != struct.pack('<Bi', Types.TbInt32.value, 2):
                raise ValueError("unsupported protocol version")
        except (OSError, ValueError) as exc:
            raise ValueError(
                "CAVE instance protocol v2 is unavailable. Update the MADIconnect "
                "plugin on the CAVE server and cluster nodes, then reconnect."
            ) from exc

    def _recv_exact(self, sock, n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf
    
    def receive_loop(self, sock):
        try:
            while self.connected and self.sock is sock:
                header = self._recv_exact(sock, 16)
                if not header:
                    break
                sender_id, send_type, msg_type, payload_len = struct.unpack('>iiii', header)
                payload = self._recv_exact(sock, payload_len)
                if payload is None:
                    break
                if self.sock is not sock:
                    break
    
                # optional leading byteorder marker (0/1) — handle both cases
                if payload and payload[0] in (0, 1) and len(payload) > 1 and payload[1] in (
                    Types.TbInt32.value, Types.TbString.value, Types.TbFloat.value
                ):
                    payload = payload[1:]
    
                if msg_type == MsgType.MSG_SELECT_NEURONS.value:
                    try:
                        off = 0
    
                        if payload[off] != Types.TbInt32.value:
                            raise ValueError(f"expected int tag, got {payload[off]}")
                        off += 1
                        (count,) = struct.unpack_from("<i", payload, off)
                        off += 4
    
                        names = []
                        for _ in range(count):
                            if payload[off] != Types.TbString.value:
                                raise ValueError(f"expected string tag, got {payload[off]}")
                            off += 1
                            end = payload.find(b"\x00", off)
                            if end < 0:
                                raise ValueError("unterminated string")
                            names.append(payload[off:end].decode("latin-1"))
                            off = end + 1
    
                        self.selection_signal.emit(names)
                    except Exception as e:
                        self.log_signal.emit(f"Selection decode failed: {e}")
                    continue
    
                self.log_signal.emit(f"Received msg_type={msg_type}, len={payload_len}")
        except Exception as e:
            self.log_signal.emit(f"Receive error: {e}")
        finally:
            try:
                sock.close()
            except Exception:
                pass
            if self.sock is sock:
                self.sock = None
                self.connected = False
                self.disconnected_signal.emit()
                self.log_signal.emit("Disconnected from server")

    def disconnect(self):
        if self.connected:
            self.connected = False
            sock, self.sock = self.sock, None
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            finally:
                sock.close()
            self.disconnected_signal.emit()
            self.log_signal.emit("Disconnected from server")
        else:
            self.log_signal.emit("Already disconnected")

    def send_message(self, msg_type, payload):
        if not self.connected:
            self.log_signal.emit("Not connected")
            return
        try:
            sender_id = 8564
            send_type = 0
            
            data = struct.pack('<B', 1) + payload  # Start with byte order
            payload_len = len(data)
            header = struct.pack('>iiii', sender_id, send_type, msg_type.value, payload_len)

            self.sock.sendall(header + data)
            self.log_signal.emit(f"Message {msg_type.name} sent ({payload_len} bytes)")
            return True
        except Exception as e:
            self.log_signal.emit(f"Send failed: {e}")
            self.disconnect()
            return False

    @staticmethod
    def _string_token(value):
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError("CAVE strings must be nonempty and contain no NUL")
        return bytes([Types.TbString.value]) + value.encode('latin-1') + b'\x00'

    def send_load_neurons(self, instances):
        """Send v2 load records: N followed by (instance_id, source filename)."""
        if not instances:
            self.log_signal.emit("No instances provided for loading neurons")
            return
        data = struct.pack('<Bi', Types.TbInt32.value, len(instances))
        seen = set()
        for record in instances:
            if not isinstance(record, (tuple, list)) or len(record) != 2:
                raise ValueError("CAVE loads require (instance_id, filename) pairs")
            instance_id, filename = record
            data += self._string_token(instance_id) + self._string_token(filename)
            if instance_id in seen:
                raise ValueError("Duplicate CAVE instance ID in load batch")
            seen.add(instance_id)
        return self.send_message(MsgType.MSG_LOAD_NEURONS, data)

    def send_show_neurons(self, instance_ids):
        """Sends a message to show neurons."""
        if instance_ids is None or len(instance_ids) == 0:
            self.log_signal.emit("No instance_ids provided for showing neurons")
            return  # No instance_ids provided, nothing to send
        data = b''
        data += struct.pack('<B', Types.TbInt32.value)  # Array length type
        data += struct.pack('<i', len(instance_ids))  # Array length
        for instance_id in instance_ids:
            data += self._string_token(instance_id)
        return self.send_message(MsgType.MSG_SHOW_NEURONS, data)

    def send_hide_neurons(self, instance_ids):
        """Sends a message to hide neurons."""
        if instance_ids is None or len(instance_ids) == 0:
            self.log_signal.emit("No instance_ids provided for hiding neurons")
            return  # No instance_ids provided, nothing to send
        data = b''
        data += struct.pack('<B', Types.TbInt32.value)  # Array length type
        data += struct.pack('<i', len(instance_ids))  # Array length
        for instance_id in instance_ids:
            data += self._string_token(instance_id)
        return self.send_message(MsgType.MSG_HIDE_NEURONS, data)
    
    def send_color_neurons_rgb(self, instance_ids, r_i: int, g_i: int, b_i: int, transparency: float = 0.0):
        """Send MSG_COLOR_NEURONS as: int r, int g, int b, float transparency, int N, N×string."""
        if not self.connected:
            self.log_signal.emit("Not connected")
            return
        if not instance_ids:
            self.log_signal.emit("No instance_ids provided for coloring neurons")
            return
    
        # uses existing Types + MsgType
    
        # clamp/convert
        r_i = max(0, min(255, int(r_i)))
        g_i = max(0, min(255, int(g_i)))
        b_i = max(0, min(255, int(b_i)))
        transparency = float(max(0.0, min(1.0, transparency)))
    
        data = b""
        data += struct.pack('<B', Types.TbInt32.value) + struct.pack('<i', r_i)
        data += struct.pack('<B', Types.TbInt32.value) + struct.pack('<i', g_i)
        data += struct.pack('<B', Types.TbInt32.value) + struct.pack('<i', b_i)
        data += struct.pack('<B', Types.TbFloat.value)  + struct.pack('<f', transparency)
        data += struct.pack('<B', Types.TbInt32.value) + struct.pack('<i', len(instance_ids))
        for instance_id in instance_ids:
            data += self._string_token(instance_id)
    
        return self.send_message(MsgType.MSG_COLOR_NEURONS, data)


    def send_load_volume(self):
        """Sends a message to load the volume."""
        data = b''  # No payload for load volume
        self.send_message(MsgType.MSG_LOAD_VOLUME, data)

    def send_show_volume(self):
        """Sends a message to show the volume."""
        data = b''  # No payload for show volume
        self.send_message(MsgType.MSG_SHOW_VOLUME, data)
    
    def send_hide_volume(self):
        """Sends a message to hide the volume."""
        data = b''  # No payload for hide volume
        self.send_message(MsgType.MSG_HIDE_VOLUME, data)

    def send_view_all(self):
        """Sends a view all message to the server."""
        data = b''  # No payload for view all
        self.send_message(MsgType.MSG_VIEW_ALL, data)

    def send_test_message(self):
        """Sends a test message to the server."""
        data = (    
            struct.pack('<B', Types.TbInt32.value) +  # 1 byte: next up is an int32
            struct.pack('<i', 72)                     # 4 bytes: signed 32-bit int value 72
        )
        self.send_message(MsgType.MSG_TEST, data)
