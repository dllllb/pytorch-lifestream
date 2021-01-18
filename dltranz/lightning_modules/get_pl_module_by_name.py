from dltranz.lightning_modules.coles_module import CoLESModule
from dltranz.lightning_modules.cpc_module import CpcModule
from dltranz.lightning_modules.rtd_module import RtdModule
from dltranz.lightning_modules.sop_nsp_module import SopNspModule

class GetPLModuleByName(object):

    def __init__(self, pl_module_name):
        self.pl_module_name = pl_module_name

    def get_pl_module_by_name(self):
        pl_module = None
        for m in [CoLESModule, CpcModule, SopNspModule, RtdModule]:
            if m.__name__ == self.pl_module_name:
                pl_module = m
                break
        if pl_module is None:
            raise NotImplementedError(f'Unknown pl module {self.pl_module_name}')

        return pl_module
