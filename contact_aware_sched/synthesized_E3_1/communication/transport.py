"""Persistent TCP ring with dedicated TX/RX threads and bounded frame jobs.

The main training thread never issues distributed collectives. One TX
thread owns the outgoing TCP connection; one RX thread owns the incoming
connection. A frame is acknowledged only after its complete payload has
been received and validated. TCP handles retransmission within a live
connection; connection loss aborts the run, never silently replays data.
"""
import csv
import json
import math
import queue
import select
import socket
import struct
import threading
import time
from dataclasses import dataclass,field
from pathlib import Path

import numpy as np

from .contact_plan import LinkPlan


HEADER=struct.Struct('!4sQBBHII')
HELLO=struct.Struct('!4sI')
DATA=b'RING'
ACK=b'ACK!'
STOP=object()


@dataclass
class Job:
    step:int
    phase:int
    round:int
    chunk:int
    array:np.ndarray
    done:threading.Event=field(default_factory=threading.Event)
    error:object=None
    wait_s:float=0.
    payload_bytes:int=0


class RingTransport:
    def __init__(self, rank, peers, port, run_id, scheduler, *, frame_bytes=1048576,
                 margin=.5, factor=2., open_guard=.25, min_transfer=.02,
                 connect_timeout=180., io_timeout=900., drain_guard=2.,
                 drain_timeout=240.):
        self.rank=int(rank);self.peers=list(peers);self.world=len(peers)
        if self.world!=4 or not 0<=self.rank<self.world:
            raise ValueError('Ring transport requires four ranks')
        if len(set(self.peers))!=self.world or not 1<=int(port)<=65535:
            raise ValueError('Invalid ring endpoints')
        self.next=(self.rank+1)%self.world
        self.prev=(self.rank-1)%self.world
        self.port=int(port);self.run_id=str(run_id)
        self.scheduler=scheduler
        self.mode=scheduler.mode
        self.frame_bytes=int(frame_bytes)
        if not 4<=self.frame_bytes<=64*1024*1024 or self.frame_bytes%4:
            raise ValueError('Frame bytes must be a multiple of 4, between 4 B and 64 MiB')
        self.connect_timeout=float(connect_timeout);self.io_timeout=float(io_timeout)
        self.drain_guard=float(drain_guard);self.drain_timeout=float(drain_timeout)
        if (not all(math.isfinite(v) for v in (self.connect_timeout,self.io_timeout,self.drain_timeout,self.drain_guard))
                or min(self.connect_timeout,self.io_timeout,self.drain_timeout)<=0
                or not 0<=self.drain_guard<self.drain_timeout):
            raise ValueError('Invalid ring timeout')
        self.start=scheduler.start
        self.end=scheduler.end
        sats=scheduler.sats
        edges=scheduler.edges
        def plan(src,dst):
            return LinkPlan(edges[sats[src],sats[dst]],edges[sats[dst],sats[src]],
                scheduler.end-scheduler.start,factor=factor,margin=margin,
                open_guard=open_guard,min_transfer=min_transfer)
        self.tx_plan=plan(self.rank,self.next)
        self.rx_plan=plan(self.prev,self.rank)
        self.cancelled=threading.Event()
        self.txq=queue.Queue(maxsize=1);self.rxq=queue.Queue(maxsize=1)
        self.tx_sock=None;self.rx_sock=None;self.listener=None
        self.threads=[]
        self.failure=None
        self.failure_lock=threading.Lock()
        self.log_lock=threading.Lock()
        self.fp=(Path(scheduler.output)/f'e3_ring_transfers_rank{self.rank}.csv').open('w',newline='',buffering=65536)
        self.writer=csv.DictWriter(self.fp,fieldnames=[
            'rank','peer','direction','mode','step','phase','round','chunk',
            'ready_unix','first_frame_unix','complete_unix','payload_bytes',
            'frames','contact_wait_s','io_s','ack_or_receive_after_window',
            'outside_contact_frames','post_horizon_frames','status','error'])
        self.writer.writeheader()
        self.last_collective_wait_s=0.
        self.last_collective_tx_bytes=0
        self.last_collective_rx_bytes=0

    def set_epoch(self,start,duration):
        self.start=float(start);self.end=self.start+float(duration)
        if abs(duration-self.tx_plan.duration)>1e-6:
            raise ValueError('Trace duration changed after socket initialization')

    @staticmethod
    def _configure(sock):
        sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        sock.settimeout(.25)

    def connect(self):
        """Set up both sockets and validate peer identity before tc replay."""
        deadline=time.monotonic()+self.connect_timeout
        listener=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.listener=listener
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind((self.peers[self.rank],self.port))
        listener.listen(2);listener.settimeout(.25)
        try:
            while time.monotonic()<deadline:
                sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                sock.settimeout(1.)
                try:
                    sock.bind((self.peers[self.rank],0))
                    sock.connect((self.peers[self.next],self.port))
                    self._configure(sock)
                    self.tx_sock=sock
                    break
                except OSError:
                    sock.close()
                    time.sleep(.1)
            else:
                raise TimeoutError('Ring TX startup timed out')
            # Tiny run-scoped handshake, before any traffic is shaped.
            payload=json.dumps({'run_id':self.run_id,'rank':self.rank,
                                'frame_bytes':self.frame_bytes}).encode()
            self._send(self.tx_sock,HELLO.pack(b'HELO',len(payload))+payload,
                       startup_deadline=deadline)
            while time.monotonic()<deadline:
                try:sock,addr=listener.accept()
                except socket.timeout:continue
                if addr[0]!=self.peers[self.prev]:
                    sock.close()
                    continue
                self.rx_sock=sock
                self._configure(sock)
                break
            else:
                raise TimeoutError('Ring RX startup timed out')
            magic,n=HELLO.unpack(self._recv(self.rx_sock,HELLO.size,startup_deadline=deadline))
            if magic!=b'HELO' or not 0<n<=4096:
                raise RuntimeError('Invalid ring handshake')
            hello=json.loads(self._recv(self.rx_sock,n,startup_deadline=deadline))
            if hello!={'run_id':self.run_id,'rank':self.prev,'frame_bytes':self.frame_bytes}:
                raise RuntimeError('Ring peer/run/frame configuration mismatch')
            listener.close();self.listener=None
            # Threads start before replay, but wait for jobs from training.
            self.threads=[threading.Thread(target=self._worker,args=('tx',),
                           name=f'ring-tx-{self.rank}',daemon=True),
                          threading.Thread(target=self._worker,args=('rx',),
                           name=f'ring-rx-{self.rank}',daemon=True)]
            for t in self.threads:t.start()
        except BaseException:
            self.close()
            raise

    def _check(self, last_progress, startup_deadline=None):
        if self.cancelled.is_set():
            raise RuntimeError('Ring transport cancelled')
        if startup_deadline is not None:
            if time.monotonic()>=startup_deadline:raise TimeoutError('Ring handshake timed out')
            return
        if time.time()>=self.end+self.drain_timeout:
            raise TimeoutError('Ring horizon drain timed out')
        # AWARE receive/ACK waits can span predicted outages. Do not apply
        # an arbitrary idle timeout to a valid planned contact wait.
        if self.mode=='off' and time.monotonic()-last_progress>self.io_timeout:
            raise TimeoutError('Reactive TCP transfer made no progress')

    def _send(self,sock,data,startup_deadline=None):
        view=memoryview(data).cast('B')
        offset=0;last=time.monotonic()
        while offset<len(view):
            self._check(last,startup_deadline)
            try:n=sock.send(view[offset:])
            except socket.timeout:continue
            if n<=0:raise ConnectionError('Ring peer closed during send')
            offset+=n;last=time.monotonic()

    def _recv(self,sock,size,startup_deadline=None):
        data=bytearray(size);view=memoryview(data)
        offset=0;last=time.monotonic()
        while offset<size:
            self._check(last,startup_deadline)
            try:n=sock.recv_into(view[offset:])
            except socket.timeout:continue
            if n<=0:raise ConnectionError('Ring peer closed during receive')
            offset+=n;last=time.monotonic()
        return data

    def _wait_slot(self,n,job):
        ready=time.monotonic();reported=False
        while True:
            self._check(ready)
            now=time.time()
            if now>=self.end:
                # The trace stops and tc agents clean up. Drain the in-flight
                # collective/terminal vote outside the measured horizon only.
                target=self.end+self.drain_guard
                if now<target:
                    self.cancelled.wait(min(.1,target-now));continue
                return time.monotonic()-ready,None,True
            if self.mode=='off':
                return 0.,None,False
            slot=self.tx_plan.next_slot(now-self.start,n)
            if slot is not None and self.start+slot[0]<=now+1e-6:
                return time.monotonic()-ready,self.start+slot[1],False
            target=self.start+slot[0] if slot else self.end+self.drain_guard
            if not reported:
                print(f'[RING TX r{self.rank}->{self.next}] WAIT_CONTACT step={job.step} '
                      f'phase={job.phase} round={job.round} next_elapsed={target-self.start:.3f}',
                      flush=True)
                reported=True
            self.cancelled.wait(min(.1,max(.001,target-now)))

    def _await_rx_slot(self,n,job):
        """Use the incoming contact plan to arm RX. Read already-buffered
        packets immediately even if the plan says closed; do not withhold
        TCP/application ACKs merely because a frame arrived late."""
        ready=time.monotonic()
        if self.mode=='off':return 0.
        reported=False
        while time.time()<self.end:
            now=time.time()
            slot=self.rx_plan.next_slot(now-self.start,n)
            target=self.start+slot[0] if slot else self.end
            if target<=now+1e-6:return time.monotonic()-ready
            if not reported:
                print(f'[RING RX r{self.rank}<-{self.prev}] WAIT_INCOMING step={job.step} '
                      f'phase={job.phase} round={job.round} next_elapsed={target-self.start:.3f}',
                      flush=True)
                reported=True
            self._check(ready)
            readable,_,_=select.select([self.rx_sock],[],[],min(.25,max(.001,target-now)))
            if readable:return time.monotonic()-ready
        return time.monotonic()-ready

    def _transfer(self,direction,job):
        ready=time.time();first=None;wait=0.;io=0.;frames=0
        after=outside=drain=0;payload=0;status='completed';error=''
        raw=memoryview(job.array).cast('B')
        total=len(raw)
        try:
            for off in range(0,total,self.frame_bytes):
                n=min(self.frame_bytes,total-off)
                if direction=='tx':
                    waited,contact_end,post=self._wait_slot(n,job)
                    wait+=waited
                    stamp=time.time()
                    if first is None:first=stamp
                    if stamp>=self.end or post:drain+=1
                    elif not self.tx_plan.containing(stamp-self.start):outside+=1
                    header=HEADER.pack(DATA,job.step,job.phase,job.round,job.chunk,off,n)
                    t=time.monotonic()
                    self._send(self.tx_sock,header)
                    self._send(self.tx_sock,raw[off:off+n])
                    reply=self._recv(self.tx_sock,HEADER.size)
                    if HEADER.unpack(reply)!=(ACK,job.step,job.phase,job.round,job.chunk,off,n):
                        raise RuntimeError('Unexpected frame ACK (sequence mismatch)')
                    io+=time.monotonic()-t
                    if contact_end is not None and time.time()>contact_end:after+=1
                else:
                    # Arm reception as soon as the dependency is known.
                    # Never stop reading an already in-flight TCP frame when
                    # a contact closes: that would create an artificial deadlock.
                    wait+=self._await_rx_slot(n,job)
                    t=time.monotonic()
                    header=self._recv(self.rx_sock,HEADER.size)
                    stamp=time.time()
                    if first is None:first=stamp
                    if HEADER.unpack(header)!=(DATA,job.step,job.phase,job.round,job.chunk,off,n):
                        raise RuntimeError('Unexpected ring frame (collective/phase/offset mismatch)')
                    raw[off:off+n]=self._recv(self.rx_sock,n)
                    self._send(self.rx_sock,HEADER.pack(ACK,job.step,job.phase,job.round,job.chunk,off,n))
                    io+=time.monotonic()-t
                    if stamp>=self.end:drain+=1
                    else:
                        window=self.rx_plan.containing(stamp-self.start)
                        if window is None:outside+=1
                        elif time.time()>self.start+window[1]:after+=1
                frames+=1;payload+=n
        except BaseException as exc:
            status='failed';error=repr(exc)
            raise
        finally:
            job.wait_s=wait;job.payload_bytes=payload
            with self.log_lock:
                self.writer.writerow(dict(rank=self.rank,peer=self.next if direction=='tx' else self.prev,
                    direction=direction,mode=self.mode,step=job.step,phase=job.phase,round=job.round,
                    chunk=job.chunk,ready_unix=ready,first_frame_unix=first,
                    complete_unix=time.time(),payload_bytes=payload,frames=frames,
                    contact_wait_s=wait,io_s=io,ack_or_receive_after_window=after,
                    outside_contact_frames=outside,post_horizon_frames=drain,status=status,error=error))
                # One flush per chunk (not per frame); result visibility
                # is diagnostic only and never a live scheduling dependency.
                self.fp.flush()

    def _worker(self,direction):
        q=self.txq if direction=='tx' else self.rxq
        while not self.cancelled.is_set():
            try:job=q.get(timeout=.1)
            except queue.Empty:continue
            if job is STOP:return
            try:self._transfer(direction,job)
            except BaseException as exc:
                job.error=exc
                with self.failure_lock:
                    if self.failure is None:self.failure=exc
                self.cancelled.set()
                self._shutdown_sockets()
            finally:job.done.set()

    def exchange(self,step,phase,round_,send_chunk,receive_chunk,array):
        """Post both directions simultaneously; training owns dependencies."""
        if self.failure is not None:raise RuntimeError('Ring worker failed') from self.failure
        if self.cancelled.is_set():raise RuntimeError('Ring transport closed')
        if array.dtype!=np.float32 or not array.flags.c_contiguous:
            raise ValueError('Ring exchanges contiguous float32 chunks')
        tx=Job(step,phase,round_,send_chunk,array)
        rx=Job(step,phase,round_,receive_chunk,np.empty_like(array))
        self.rxq.put_nowait(rx);self.txq.put_nowait(tx)
        for job in (rx,tx):
            while not job.done.wait(.1):
                if self.failure is not None:raise RuntimeError('Ring worker failed') from self.failure
                if self.cancelled.is_set():raise RuntimeError('Ring transport closed')
                if time.time()>=self.end+self.drain_timeout:
                    raise TimeoutError('Ring exchange exceeded drain deadline')
            if job.error is not None:raise RuntimeError('Ring transfer failed') from job.error
        self.last_collective_wait_s+=tx.wait_s
        self.last_collective_tx_bytes+=tx.payload_bytes
        self.last_collective_rx_bytes+=rx.payload_bytes
        return rx.array

    def _shutdown_sockets(self):
        for sock in (self.tx_sock,self.rx_sock,self.listener):
            if sock is not None:
                try:sock.shutdown(socket.SHUT_RDWR)
                except OSError:pass

    def close(self):
        self.cancelled.set()
        # Normal close follows the final fully acknowledged exchange. No
        # live work may be submitted afterwards; failed work is aborted.
        self._shutdown_sockets()
        for t in self.threads:t.join(timeout=2)
        for sock in (self.tx_sock,self.rx_sock,self.listener):
            if sock is not None:
                try:sock.close()
                except OSError:pass
        if not any(t.is_alive() for t in self.threads) and not self.fp.closed:
            self.fp.close()
