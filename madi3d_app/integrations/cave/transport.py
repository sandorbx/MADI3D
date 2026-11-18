"""Event-driven CAVE transport. All methods run on the owning Qt thread."""

import struct
from collections import deque
from enum import Enum

from PySide6 import QtCore, QtNetwork

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

class _Connection(QtCore.QObject):
    """One generation owns its socket, deadlines and complete pending frames."""

    event = QtCore.Signal(int, str, object)
    READ_CHUNK_BYTES = 64 * 1024
    MAX_PAYLOAD_BYTES = 8 * 1024 * 1024  # Selection lists are the largest incoming messages.

    def __init__(self, generation, *, connect_timeout_ms, send_timeout_ms,
                 max_pending_bytes, max_pending_items, parent):
        super().__init__(parent)
        self.generation = generation
        self.max_pending_bytes = max_pending_bytes
        self.max_pending_items = max_pending_items
        self.send_timeout_ms = send_timeout_ms
        self.phase = "connecting"
        self.pending = deque()
        self.pending_bytes = 0  # includes the complete in-flight frame
        self.inflight = None
        self.remaining = 0
        self.control_remaining = 0
        self.input = bytearray()
        self._read_scheduled = False
        self._eof_reason = None
        self.socket = QtNetwork.QTcpSocket(self)
        self.socket.setReadBufferSize(self.READ_CHUNK_BYTES)
        self.socket.connected.connect(self._tcp_connected)
        self.socket.readyRead.connect(self._schedule_read)
        self.socket.bytesWritten.connect(self._written)
        self.socket.disconnected.connect(lambda: self._peer_closed("Server disconnected"))
        self.socket.errorOccurred.connect(self._socket_error)
        self.connect_timer = QtCore.QTimer(self)
        self.connect_timer.setSingleShot(True)
        self.connect_timer.timeout.connect(lambda: self._fail(
            "Connection/negotiation timed out. Update the MADIconnect plugin "
            "on the CAVE server and cluster nodes if protocol v2 is unavailable."))
        self.connect_timer.setInterval(connect_timeout_ms)
        self.send_timer = QtCore.QTimer(self)
        self.send_timer.setSingleShot(True)
        self.send_timer.timeout.connect(lambda: self._fail(
            "Send deadline exceeded; connection retired (remote completion unknown)"))

    def start(self, host, port):
        self.connect_timer.start()
        self.socket.connectToHost(host, port)

    def _tcp_connected(self):
        self.phase = "handshake"

    def _socket_error(self, error):
        if error == QtNetwork.QAbstractSocket.SocketError.RemoteHostClosedError:
            self._peer_closed(self.socket.errorString())
        else:
            self._fail(self.socket.errorString())

    def _peer_closed(self, reason):
        if self.phase == "retired":
            return
        self._eof_reason = reason
        self.send_timer.stop()
        # Qt can announce EOF before our scheduled read. Forbid further writes,
        # but drain complete buffered frames before discarding a truncated tail.
        self._schedule_read()

    def retire(self):
        if self.phase == "retired":
            return
        self.phase = "retired"
        self.connect_timer.stop()
        self.send_timer.stop()
        self.pending.clear()
        self.pending_bytes = 0
        self.inflight = None
        self.remaining = 0
        self.input.clear()
        # abort discards Qt's write buffer too; a possibly partial frame is never
        # followed by another frame on this connection, or replayed on a new one.
        self.socket.abort()
        self.deleteLater()

    def _fail(self, reason):
        if self.phase == "retired":
            return
        was_ready = self.phase == "ready"
        self.retire()
        self.event.emit(self.generation, "failed", (was_ready, reason))

    def enqueue(self, kind, frame, replacement_key):
        if self.phase != "ready" or self._eof_reason is not None:
            return False
        # Only complete, not-yet-started color messages are replaceable. Every
        # non-color command is an ordering barrier. Append replacements at the
        # tail so intervening overlapping color batches retain their order.
        replaced = set()
        if replacement_key is not None:
            for index in range(len(self.pending) - 1, -1, -1):
                if self.pending[index][2] is None:
                    break
                if self.pending[index][2] == replacement_key:
                    replaced.add(index)
        pending = [entry for index, entry in enumerate(self.pending) if index not in replaced]
        pending.append((kind, frame, replacement_key))
        sizes = [len(entry[1]) for entry in pending]
        if self.inflight is not None:
            sizes.append(len(self.inflight[1]))
        total = sum(sizes)
        # QTcpSocket keeps a native copy of the active frame until bytesWritten.
        # Reserve the largest such copy across the queued sequence as well as
        # every retained Python frame. Only one frame may enter Qt at a time.
        if total + max(sizes) > self.max_pending_bytes or len(sizes) > self.max_pending_items:
            self.event.emit(self.generation, "overload",
                            "CAVE queue full; command not queued. Retry after pending writes finish.")
            return False
        self.pending = deque(pending)
        self.pending_bytes = total
        self.event.emit(self.generation, "queued", self._message_info(kind, frame))
        self._pump()
        return True

    def _pump(self):
        if (self.phase != "ready" or self._eof_reason is not None or self.control_remaining or
                self.inflight is not None or not self.pending):
            return
        kind, frame, _key = self.pending.popleft()
        self.inflight = (kind, frame)
        self.remaining = len(frame)
        # Absolute per-frame deadline; incremental progress never extends it.
        self.send_timer.start(self.send_timeout_ms)
        accepted = self.socket.write(frame)
        if accepted != len(frame):
            self._fail("Socket did not accept complete frame; connection retired")

    def _written(self, count):
        control = min(count, self.control_remaining)
        self.control_remaining -= control
        count -= control
        if not count:
            self._pump()
            return
        if self.phase != "ready" or self.inflight is None:
            return
        self.remaining -= count
        if self.remaining > 0:
            return
        info = self._message_info(*self.inflight)
        self.pending_bytes -= len(self.inflight[1])
        self.inflight = None
        self.send_timer.stop()
        self.event.emit(self.generation, "written", info)
        # Yield between frames, and never start a later frame from inside write.
        QtCore.QTimer.singleShot(0, self._pump)

    @staticmethod
    def _message_info(kind, frame):
        # Load N follows the 16-byte header, byte-order marker and int32 tag.
        count = struct.unpack_from('<i', frame, 18)[0] if kind == MsgType.MSG_LOAD_NEURONS else 0
        return kind, count

    def _schedule_read(self):
        if not self._read_scheduled and self.phase != "retired":
            self._read_scheduled = True
            QtCore.QTimer.singleShot(0, self._read)

    def _read(self):
        self._read_scheduled = False
        if self.phase == "retired":
            return
        try:
            self._read_chunk()
        except MemoryError:
            self._fail("Insufficient memory receiving CAVE message")

    def _read_chunk(self):
        budget = self.READ_CHUNK_BYTES
        if self.phase == "handshake" and self.socket.bytesAvailable():
            if bytes(self.socket.read(1)) != b"\x01":
                self._fail("Unexpected byte order handshake")
                return
            budget -= 1
            self.phase = "negotiating"
            negotiation = b"\x01" + MessageClient._frame(
                MsgType.MSG_PROTOCOL_VERSION, struct.pack('<Bi', Types.TbInt32.value, 2))
            self.control_remaining = len(negotiation)
            self.socket.write(negotiation)
        frames = 0
        while self.phase in ("negotiating", "ready") and budget and frames < 64:
            if len(self.input) < 16:
                # Read only the header until its declared body length is accepted.
                header = bytes(self.socket.read(min(16 - len(self.input), budget)))
                self.input.extend(header)
                budget -= len(header)
                if len(self.input) < 16:
                    break
            _, _, kind, size = struct.unpack_from('>iiii', self.input)
            if size < 0:
                self._fail("Invalid negative CAVE frame length")
                return
            if size > self.MAX_PAYLOAD_BYTES:
                self._fail("CAVE frame exceeds the incoming message size limit")
                return
            missing = 16 + size - len(self.input)
            if missing and budget:
                chunk = bytes(self.socket.read(min(missing, budget)))
                self.input.extend(chunk)
                budget -= len(chunk)
            if len(self.input) < 16 + size:
                break
            with memoryview(self.input) as frame_view:
                payload = bytes(frame_view[16:16 + size])
            self.input.clear()
            if self.phase == "negotiating":
                if size == 6 and payload[0] in (0, 1):
                    payload = payload[1:]
                if kind != MsgType.MSG_PROTOCOL_VERSION.value or payload != struct.pack('<Bi', 9, 2):
                    self._fail("CAVE instance protocol v2 is unavailable. Update the MADIconnect "
                               "plugin on the CAVE server and cluster nodes, then reconnect.")
                    return
                self.connect_timer.stop()
                self.phase = "ready"
                self.event.emit(self.generation, "connected", None)
                self._pump()
            else:
                self.event.emit(self.generation, "received", (kind, payload))
            frames += 1
        if self.phase != "retired" and self.socket.bytesAvailable():
            self._schedule_read()
        elif self._eof_reason is not None and not self._read_scheduled:
            self._fail(self._eof_reason)


class MessageClient(QtCore.QObject):
    """Async transport/codec; True from send means queued, never acknowledged."""

    log_signal = QtCore.Signal(str)
    selection_signal = QtCore.Signal(int, list)
    disconnected_signal = QtCore.Signal()
    connecting_signal = QtCore.Signal(int, str, int)
    connected_signal = QtCore.Signal(int, str, int)
    state_changed = QtCore.Signal(str)
    queued_signal = QtCore.Signal(int, int)
    written_signal = QtCore.Signal(int, int)
    load_queued_signal = QtCore.Signal(int, int)
    load_written_signal = QtCore.Signal(int, int)

    def __init__(self, *, connect_timeout_ms=5000, send_timeout_ms=5000,
                 max_pending_bytes=4 * 1024 * 1024, max_pending_items=256):
        super().__init__()
        if min(connect_timeout_ms, send_timeout_ms, max_pending_bytes, max_pending_items) <= 0:
            raise ValueError("Transport limits must be positive")
        self._policy = dict(connect_timeout_ms=connect_timeout_ms, send_timeout_ms=send_timeout_ms,
                            max_pending_bytes=max_pending_bytes, max_pending_items=max_pending_items)
        self._connection = None
        self._generation = 0
        self._endpoint = None

    @property
    def generation(self):
        return self._generation

    @property
    def state(self):
        if self._connection is None or self._connection._eof_reason is not None:
            return "disconnected"
        return "connected" if self._connection.phase == "ready" else "connecting"

    @property
    def connected(self):
        return self.state == "connected"

    @property
    def connecting(self):
        return self.state == "connecting"

    def connect(self, host='134.95.85.134', port=33333):
        generation = self._generation + 1
        self.disconnect()
        if self._generation != generation:
            return False  # a disconnect observer started a newer request
        self._endpoint = (host, port)
        connection = _Connection(self._generation, parent=self, **self._policy)
        self._connection = connection
        connection.event.connect(self._on_event)
        self.state_changed.emit("connecting")
        if self._generation != generation or self._connection is not connection:
            return False
        self.connecting_signal.emit(generation, host, port)
        if self._generation != generation or self._connection is not connection:
            return False
        connection.start(host, port)
        return True

    def disconnect(self):
        self._generation += 1
        generation = self._generation
        connection, self._connection = self._connection, None
        if connection is None:
            return
        canceled = len(connection.pending) + (connection.inflight is not None)
        connection.retire()
        self.state_changed.emit("disconnected")
        if self._generation != generation:
            return
        self.disconnected_signal.emit()
        if canceled and self._generation == generation:
            self.log_signal.emit(f"CAVE disconnected; {canceled} pending commands canceled "
                                 "(remote completion unknown; no replay)")

    @QtCore.Slot(int, str, object)
    def _on_event(self, generation, event, value):
        if self._connection is None or generation != self._generation:
            return
        if event == "connected":
            if not self.connected:
                return
            endpoint = self._endpoint
            self.state_changed.emit("connected")
            if generation == self._generation and self.connected:
                self.connected_signal.emit(generation, *endpoint)
        elif event == "failed":
            was_ready, reason = value
            self._connection = None
            self.state_changed.emit("disconnected")
            if generation != self._generation:
                return
            self.disconnected_signal.emit()
            if generation != self._generation:
                return
            self.log_signal.emit(f"{'Transport' if was_ready else 'Connection'} failed: {reason}; "
                                 "pending commands discarded, no replay")
        elif event == "overload":
            self.log_signal.emit(value)
        elif event == "queued":
            kind, count = value
            if kind == MsgType.MSG_LOAD_NEURONS:
                self.load_queued_signal.emit(generation, count)
            if generation == self._generation:
                self.queued_signal.emit(generation, kind.value)
        elif event == "written":
            kind, count = value
            if kind == MsgType.MSG_LOAD_NEURONS:
                self.load_written_signal.emit(generation, count)
            if generation != self._generation:
                return
            self.written_signal.emit(generation, kind.value)
            if generation == self._generation:
                self.log_signal.emit(f"Message {kind.name} written locally (remote completion unconfirmed)")
        elif event == "received":
            try:
                self._receive_message(*value)
            except MemoryError:
                # Qt slot exceptions do not propagate back through event.emit.
                if self._connection is not None and generation == self._generation:
                    self._connection._fail("Insufficient memory decoding CAVE message")

    def _receive_message(self, kind, payload):
        if payload and payload[0] in (0, 1) and len(payload) > 1 and payload[1] in (9, 12, 10):
            payload = payload[1:]
        if kind != MsgType.MSG_SELECT_NEURONS.value:
            self.log_signal.emit(f"Received msg_type={kind}, len={len(payload)}")
            return
        try:
            if payload[0] != Types.TbInt32.value:
                raise ValueError("expected int tag")
            count, = struct.unpack_from('<i', payload, 1)
            if count < 0:
                raise ValueError("negative selection count")
            offset, names = 5, []
            for _ in range(count):
                if payload[offset] != Types.TbString.value:
                    raise ValueError("expected string tag")
                offset += 1
                end = payload.index(0, offset)
                names.append(payload[offset:end].decode('latin-1'))
                offset = end + 1
            self.selection_signal.emit(self._generation, names)
        except (IndexError, ValueError, struct.error) as exc:
            self.log_signal.emit(f"Selection decode failed: {exc}")

    @staticmethod
    def _frame(msg_type, payload):
        return struct.pack('>iiii', 8564, 0, msg_type.value, len(payload) + 1) + b'\x01' + payload

    def send_message(self, msg_type, payload, *, replacement_key=None):
        if not self.connected:
            self.log_signal.emit("Not connected; command not queued")
            return False
        return self._connection.enqueue(msg_type, self._frame(msg_type, payload), replacement_key)

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
        data = bytearray(struct.pack('<Bi', Types.TbInt32.value, len(instances)))
        seen = set()
        for record in instances:
            if not isinstance(record, (tuple, list)) or len(record) != 2:
                raise ValueError("CAVE loads require (instance_id, filename) pairs")
            instance_id, filename = record
            data.extend(self._string_token(instance_id))
            data.extend(self._string_token(filename))
            if instance_id in seen:
                raise ValueError("Duplicate CAVE instance ID in load batch")
            seen.add(instance_id)
        return self.send_message(MsgType.MSG_LOAD_NEURONS, bytes(data))

    def send_show_neurons(self, instance_ids):
        """Sends a message to show neurons."""
        if instance_ids is None or len(instance_ids) == 0:
            self.log_signal.emit("No instance_ids provided for showing neurons")
            return  # No instance_ids provided, nothing to send
        data = bytearray()
        data += struct.pack('<B', Types.TbInt32.value)  # Array length type
        data += struct.pack('<i', len(instance_ids))  # Array length
        for instance_id in instance_ids:
            data += self._string_token(instance_id)
        return self.send_message(MsgType.MSG_SHOW_NEURONS, bytes(data))

    def send_hide_neurons(self, instance_ids):
        """Sends a message to hide neurons."""
        if instance_ids is None or len(instance_ids) == 0:
            self.log_signal.emit("No instance_ids provided for hiding neurons")
            return  # No instance_ids provided, nothing to send
        data = bytearray()
        data += struct.pack('<B', Types.TbInt32.value)  # Array length type
        data += struct.pack('<i', len(instance_ids))  # Array length
        for instance_id in instance_ids:
            data += self._string_token(instance_id)
        return self.send_message(MsgType.MSG_HIDE_NEURONS, bytes(data))
    
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
    
        data = bytearray()
        data += struct.pack('<B', Types.TbInt32.value) + struct.pack('<i', r_i)
        data += struct.pack('<B', Types.TbInt32.value) + struct.pack('<i', g_i)
        data += struct.pack('<B', Types.TbInt32.value) + struct.pack('<i', b_i)
        data += struct.pack('<B', Types.TbFloat.value)  + struct.pack('<f', transparency)
        data += struct.pack('<B', Types.TbInt32.value) + struct.pack('<i', len(instance_ids))
        for instance_id in instance_ids:
            data += self._string_token(instance_id)
    
        return self.send_message(MsgType.MSG_COLOR_NEURONS, bytes(data),
                                 replacement_key=tuple(instance_ids))


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
