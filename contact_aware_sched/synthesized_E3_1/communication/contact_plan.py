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

    def credit_limit(self, elapsed, payload_sizes, max_frames):
        """Conservative credit budget including outstanding unacknowledged bytes.

        One RTT for a pipelined burst, serialization of every frame/ACK, and
        the configured safety factor and margin. This is an admission estimate,
        not a guarantee of wire completion. Credit is also bounded by ~2 BDP.
        """
        w=self.containing(elapsed)
        if w is None or elapsed<w[0]+self.open_guard:return 0
        _,end,rate,ack_rate,fd,rd=w
        rtt=fd+rd
        # Two bandwidth-delay products, rounded up, capped by configured memory.
        first=max(1,payload_sizes[0]) if payload_sizes else 1
        bdp_limit=max(1,math.ceil(2*rate*rtt/8/first))
        total=0
        for i,n in enumerate(payload_sizes[:min(max_frames,bdp_limit)],1):
            total+=(n+128)*8/rate+128*8/ack_rate
            if elapsed+max(self.min_transfer,self.factor*(total+rtt))+self.margin>end:
                return i-1
        return min(len(payload_sizes),max_frames,bdp_limit)

    def next_credit_slot(self, elapsed, payload_bytes):
        for w in self.windows:
            start=max(elapsed,w[0]+self.open_guard)
            if self.credit_limit(start,[payload_bytes],1):
                return start,w[1]
        return None


class RingConnectivity:
    """Classify planned bidirectional ring availability, independent of TX policy."""
    def __init__(self, edges, sats, duration):
        from bisect import bisect_right
        self._bisect=bisect_right
        self.duration=float(duration)
        pairs=[(edges[sats[i],sats[(i+1)%len(sats)]],
                edges[sats[(i+1)%len(sats)],sats[i]]) for i in range(len(sats))]
        bounds=sorted({0.,self.duration}|{t for pair in pairs for segs in pair
                         for s in segs for t in s[:2] if 0<t<self.duration})
        self.intervals=[]
        for a,b in zip(bounds,bounds[1:]):
            t=(a+b)/2
            count=sum(all(any(s[0]<=t<s[1] and s[2]==1 and s[3]>0
                              for s in segs) for segs in pair) for pair in pairs)
            state='full' if count==len(pairs) else 'partial' if count else 'none'
            if self.intervals and self.intervals[-1][2]==state:
                self.intervals[-1]=(self.intervals[-1][0],b,state)
            else:self.intervals.append((a,b,state))
        self.starts=[a for a,b,state in self.intervals]

    def interval(self,t):
        i=self._bisect(self.starts,t)-1
        if i<0 or i>=len(self.intervals):return None
        a,b,state=self.intervals[i]
        return (a,b,state) if a<=t<b else None

    def classify(self, launch, ack):
        a=self.interval(launch);b=self.interval(ack)
        if a is None or b is None:return 'outside'
        return a[2] if a==b else 'crossing'
