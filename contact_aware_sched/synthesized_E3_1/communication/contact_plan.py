"""Local bidirectional TCP contact admission for a single directed transfer.

Only the data edge and its reverse ACK edge are required. The rest of
the collective can use other contacts independently. Times are relative
to the shared trace epoch; rates already include the emulator rate scale.
"""
import math


class LinkPlan:
    def __init__(self, forward, reverse, duration, *, factor=2., margin=.5,
                 open_guard=.25, min_transfer=.02):
        self.duration=float(duration)
        self.factor=float(factor)
        self.margin=float(margin)
        self.open_guard=float(open_guard)
        self.min_transfer=float(min_transfer)
        if (not all(math.isfinite(x) for x in (self.duration,self.factor,self.margin,
                                               self.open_guard,self.min_transfer))
                or self.duration<=0 or self.factor<1
                or min(self.margin,self.open_guard,self.min_transfer)<0):
            raise ValueError('Invalid ring contact admission parameters')
        # Only adjacent segments of these two local incident links matter.
        bounds=sorted({0.,self.duration} |
                      {t for s in list(forward)+list(reverse) for t in s[:2]
                       if 0<t<self.duration})
        self.windows=[]
        for a,b in zip(bounds,bounds[1:]):
            middle=(a+b)/2
            f=next((s for s in forward if s[0]<=middle<s[1]),None)
            r=next((s for s in reverse if s[0]<=middle<s[1]),None)
            if not f or not r or f[2]!=1 or r[2]!=1 or min(f[3],r[3])<=0:
                continue
            # Conservative propagation and rates across any merged regime.
            key=(float(f[3]),float(r[3]),float(f[4]),float(r[4]))
            if not all(math.isfinite(x) and x>=0 for x in key):
                raise ValueError('Invalid contact rate/delay')
            if self.windows and abs(self.windows[-1][1]-a)<1e-8 and self.windows[-1][2:]==key:
                self.windows[-1]=(self.windows[-1][0],b,*key)
            else:
                self.windows.append((a,b,*key))

    def estimate(self, payload_bytes, window, overhead_bytes=64):
        _,_,forward_rate,reverse_rate,forward_delay,reverse_delay=window
        # Include application ACK, TCP/IP overhead allowance and one RTT.
        return max(self.min_transfer,self.factor*(
            (payload_bytes+overhead_bytes)*8/forward_rate
            + overhead_bytes*8/reverse_rate + forward_delay+reverse_delay))

    def next_slot(self, elapsed, payload_bytes):
        for w in self.windows:
            start=max(elapsed,w[0]+self.open_guard)
            estimate=self.estimate(payload_bytes,w)
            if start+estimate+self.margin<=w[1]:
                return start,w[1],estimate
        return None

    def containing(self, elapsed):
        return next((w for w in self.windows if w[0]<=elapsed<w[1]),None)
