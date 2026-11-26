"""Exact synchronous ring sum/mean with a piggybacked global stop vote."""
import time
import numpy as np


class RingAllReduce:
    def __init__(self, transport):
        self.transport=transport
        self.rank=transport.rank
        self.world=transport.world
        self.last_step=None

    def reduce_mean(self, vector, step, stop=None):
        """Return (mean, stop). No optimizer may apply a terminal vote.

        All ranks append one scalar stop vote. It is summed with the data,
        so every rank makes the same termination decision without an
        out-of-band per-step barrier. A final vote may drain after replay.
        """
        if self.last_step is not None and step<=self.last_step:
            raise ValueError('Collective step must increase')
        self.last_step=step
        x=np.asarray(vector,dtype=np.float32).reshape(-1)
        if x.size<1:raise ValueError('Empty collective')
        stop=bool(time.time()>=self.transport.end if stop is None else stop)
        logical_size=x.size
        n=logical_size+1
        chunk_size=(n+self.world-1)//self.world
        padded=np.zeros(chunk_size*self.world,dtype=np.float32)
        padded[:logical_size]=x
        padded[logical_size]=float(stop)
        chunks=[padded[i*chunk_size:(i+1)*chunk_size] for i in range(self.world)]
        self.transport.last_collective_wait_s=0.
        self.transport.last_collective_tx_bytes=0
        self.transport.last_collective_rx_bytes=0
        # Reduce-scatter: after P-1 rounds rank r owns chunk (r+1)%P.
        for i in range(self.world-1):
            send=(self.rank-i)%self.world
            receive=(self.rank-i-1)%self.world
            incoming=self.transport.exchange(step,0,i,send,receive,chunks[send])
            chunks[receive]+=incoming
        # All-gather: redistribute the fully reduced chunks.
        for i in range(self.world-1):
            send=(self.rank+1-i)%self.world
            receive=(self.rank-i)%self.world
            incoming=self.transport.exchange(step,1,i,send,receive,chunks[send])
            chunks[receive][:]=incoming
        global_stop=bool(padded[logical_size]>0)
        if global_stop:
            return None,True
        result=padded[:logical_size].copy()
        result/=self.world
        return result,False
