from . simul_trans_text_agent import SimulTransTextAgent
from . import register_agent

@register_agent("simul_trans_text_fast")
class SimulTransTextAgentFast(SimulTransTextAgent):

    def load_model(self, args):
        super().load_model(args)
        import torch
        self.use_cuda = False#torch.cuda.is_available()
        if self.use_cuda:
            self.model.cuda()
            self.tensor = torch.cuda
        else:
            self.tensor = torch
        self.torch = torch

    def init_states(self):
        states = super().init_states()
        states["indices"]["tgt"] = self.tensor.LongTensor(
            [[self.model.decoder.dictionary.eos()]]
        )
        return states

    def _append_indices(self, states, new_indices, key):
        indices = self.tensor.LongTensor([new_indices])
        if len(states["indices"][key]) == 0:
            states["indices"][key] = indices
        else:
            states["indices"][key] = self.torch.cat(
                [states["indices"][key], indices],
                dim=1
            )