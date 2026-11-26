"""Gossip policy: admit only point-to-point snapshot sends.

No global barriers are permitted at epoch boundaries. Each rank evaluates
its own current weights and uses only the outgoing contact to its ring peer.
"""
from e3_schedule import ContactScheduler as _Base

class ContactScheduler(_Base):
    def gate(self,scope,step,bytes_,**kwargs):
        if scope != 'peer':
            raise ValueError('Gossip scheduling must remain per-peer, never a global collective')
        return super().gate(scope,step,bytes_,**kwargs)
