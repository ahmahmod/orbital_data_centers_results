"""Frontend TCP control plane; no live NFS control-file polling."""
import hmac
import http.client
import json
import os
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from e3_startup import atomic_json

_KEY=re.compile(r'(?:nccl|reactive_stop|replay|failure|startup)/[A-Za-z0-9_./-]+$')

def valid_key(key):
    return isinstance(key,str) and bool(_KEY.fullmatch(key)) and all(
        part not in ('','.','..') for part in key.split('/'))

class Broker:
    def __init__(self, host, token, output):
        self.data={}
        self.lock=threading.Lock()
        output=Path(output)
        broker=self
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def setup(self):
                super().setup()
                self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                self.connection.settimeout(30)
            def log_message(self,*_):pass
            def reply(self, code, value):
                payload=json.dumps(value,separators=(',',':')).encode()
                self.send_response(code)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            def do_POST(self):
                if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                    self.close_connection=True
                    self.reply(403,{'error':'forbidden'})
                    return
                try:
                    n=int(self.headers.get('Content-Length','0'))
                    if not 0<n<=262144:raise ValueError('invalid request size')
                    body=json.loads(self.rfile.read(n))
                    if self.path=='/read':
                        keys=body['keys']
                        if not isinstance(keys,list) or len(keys)>128 or not all(map(valid_key,keys)):
                            raise ValueError('invalid keys')
                        with broker.lock:value=[broker.data.get(k) for k in keys]
                    elif self.path=='/write':
                        key,value=body['key'],body['value']
                        if not valid_key(key) or not isinstance(value,dict):raise ValueError('invalid message')
                        with broker.lock:
                            # Frontend mirrors diagnostic grant/stop records, not replay polling.
                            if key.startswith(('nccl/','reactive_stop/')):
                                atomic_json(output/'.e3_control'/key,value)
                            broker.data[key]=value
                        value=True
                    elif self.path=='/ping':value=True
                    else:
                        self.reply(404,{'error':'unknown operation'})
                        return
                    self.reply(200,value)
                except (KeyError,TypeError,ValueError) as exc:self.reply(400,{'error':str(exc)})
                except OSError as exc:self.reply(500,{'error':str(exc)})
        self.server=ThreadingHTTPServer((host,0),Handler)
        self.server.daemon_threads=True
        self.endpoint=f'http://{host}:{self.server.server_port}'
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

class ControlClient:
    def __init__(self,endpoint,token):
        url=urlsplit(endpoint)
        if url.scheme!='http' or not url.hostname or not url.port:raise ValueError('invalid broker endpoint')
        self.host,self.port,self.token=url.hostname,url.port,token
        self.connection=None
    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection=None
    def call(self,path,body):
        raw=json.dumps(body,separators=(',',':')).encode()
        for attempt in range(3):
            try:
                if self.connection is None:
                    self.connection=http.client.HTTPConnection(self.host,self.port,timeout=5)
                self.connection.request('POST',path,raw,headers={
                    'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
                response=self.connection.getresponse()
                payload=json.loads(response.read())
                if response.status!=200:raise RuntimeError(f'E3 broker {response.status}: {payload}')
                return payload
            except (OSError,http.client.HTTPException):
                self.close()
                if attempt==2:raise
    def ping(self):return self.call('/ping',{})
    def read_many(self,keys):return self.call('/read',{'keys':list(keys)})
    def read(self,key):return self.read_many([key])[0]
    def write(self,key,value):return self.call('/write',{'key':key,'value':value})

def client_from_env():
    endpoint,token=os.getenv('E3_CONTROL_ENDPOINT'),os.getenv('E3_CONTROL_TOKEN')
    if bool(endpoint)!=bool(token):raise RuntimeError('incomplete E3 broker configuration')
    return ControlClient(endpoint,token) if endpoint else None
