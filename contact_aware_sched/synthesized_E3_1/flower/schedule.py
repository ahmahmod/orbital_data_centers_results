"""Flower policy: gate FedAvg client upload after its local training.

Flower gRPC return is not equivalent to confirmed network completion.
All-client readiness via shared control plane is retained for E3_1.
"""
from e3_schedule import ContactScheduler as _Base

class ContactScheduler(_Base):
    def gate(self,scope,step,bytes_,**kwargs):
        if scope != 'flower':
            raise ValueError('Flower only schedules FedAvg round exchanges')
        return super().gate(scope,step,bytes_,**kwargs)
