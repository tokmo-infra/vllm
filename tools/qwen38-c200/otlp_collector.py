# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Task-private native OTLP receiver. No model, scheduler, or KV instrumentation."""

import concurrent.futures
import json
import os
import pathlib
import queue
import signal
import sqlite3
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import grpc
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2 as pb
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2_grpc as rpc

R = pathlib.Path(os.environ["QWEN38_TRACE_DIR"])
HOST = os.environ["QWEN38_TRACE_HOST"]
q = queue.Queue(100000)
stop = threading.Event()
stats = {
    "received_spans": 0,
    "written_spans": 0,
    "dropped_spans": 0,
    "export_calls": 0,
    "writer_error": None,
}


def attrs(a):
    return {
        x.key: (
            x.value.string_value
            if x.value.HasField("string_value")
            else x.value.int_value
            if x.value.HasField("int_value")
            else x.value.double_value
            if x.value.HasField("double_value")
            else x.value.bool_value
            if x.value.HasField("bool_value")
            else str(x.value)
        )
        for x in a
    }


def writer():
    con = sqlite3.connect(R / "traces.db")
    con.execute("pragma journal_mode=WAL")
    con.execute("pragma busy_timeout=5000")
    con.execute(
        "create table if not exists spans(trace_id text,span_id text,"
        "parent_span_id text,name text,start_ns integer,end_ns integer,"
        "request_id text,service text,resource_json text,payload_zlib blob,"
        "primary key(trace_id,span_id))"
    )
    con.execute("create index if not exists spans_by_request on spans(request_id)")
    con.execute("create index if not exists spans_by_time on spans(end_ns)")
    con.commit()
    while not stop.is_set() or not q.empty():
        batch = []
        try:
            batch.append(q.get(timeout=1))
        except queue.Empty:
            continue
        while len(batch) < 1024:
            try:
                batch.append(q.get_nowait())
            except queue.Empty:
                break
        try:
            con.executemany(
                "insert or ignore into spans values(?,?,?,?,?,?,?,?,?,?)", batch
            )
            con.commit()
            stats["written_spans"] += len(batch)
        except Exception as e:
            stats["writer_error"] = repr(e)
            stats["dropped_spans"] += len(batch)
    con.close()
    (R / "trace-collector-final.json").write_text(json.dumps(stats, indent=2))


class Receiver(rpc.TraceServiceServicer):
    def Export(self, request, context):
        stats["export_calls"] += 1
        for resource in request.resource_spans:
            resource_attrs = attrs(resource.resource.attributes)
            service = resource_attrs.get("service.name", "unknown")
            rj = json.dumps(resource_attrs, separators=(",", ":"))
            for scope in resource.scope_spans:
                for s in scope.spans:
                    stats["received_spans"] += 1
                    a = attrs(s.attributes)
                    rid = next(
                        (
                            str(v)
                            for k, v in a.items()
                            if "request" in k.lower() and "id" in k.lower()
                        ),
                        "",
                    )
                    row = (
                        s.trace_id.hex(),
                        s.span_id.hex(),
                        s.parent_span_id.hex(),
                        s.name,
                        s.start_time_unix_nano,
                        s.end_time_unix_nano,
                        rid,
                        service,
                        rj,
                        zlib.compress(s.SerializeToString(), 3),
                    )
                    try:
                        q.put_nowait(row)
                    except queue.Full:
                        stats["dropped_spans"] += 1
        return pb.ExportTraceServiceResponse()


class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        data = json.dumps({**stats, "pending_spans": q.qsize()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


worker = threading.Thread(target=writer)
worker.start()
server = grpc.server(concurrent.futures.ThreadPoolExecutor(8))
rpc.add_TraceServiceServicer_to_server(Receiver(), server)
assert server.add_insecure_port(f"{HOST}:18430")
server.start()
http = ThreadingHTTPServer((HOST, 18431), Health)
threading.Thread(target=http.serve_forever, daemon=True).start()


def end(*_):
    server.stop(5)
    stop.set()


signal.signal(signal.SIGTERM, end)
signal.signal(signal.SIGINT, end)
while not stop.wait(1):
    pass
worker.join(timeout=30)
