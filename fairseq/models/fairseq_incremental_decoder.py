# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

from . import FairseqDecoder


class FairseqIncrementalDecoder(FairseqDecoder):
    """Base class for incremental decoders."""

    def __init__(self):
        super().__init__()
        self._is_incremental_eval = False
        self._incremental_state = {}

    def forward(self, tokens, encoder_out):
        raise NotImplementedError

    def incremental_forward(self, tokens, encoder_out):
        """Forward pass for one time step."""
        # keep only the last token for incremental forward pass
        return self.forward(tokens[:, -1:], encoder_out)

    def incremental_inference(self):
        """Context manager for incremental inference.

        This provides an optimized forward pass for incremental inference
        (i.e., it predicts one time step at a time). If the input order changes
        between time steps, call reorder_incremental_state to update the
        relevant buffers. To generate a fresh sequence, first call
        clear_incremental_state.

        Usage:
        ```
        with model.decoder.incremental_inference():
            for step in range(maxlen):
                out, _ = model.decoder.incremental_forward(
                    tokens[:, :step], encoder_out)
                probs = torch.nn.functional.log_softmax(out[:, -1, :])
        ```
        """
        class IncrementalInference(object):
            def __init__(self, decoder):
                self.decoder = decoder

            def __enter__(self):
                self.decoder.incremental_eval(True)

            def __exit__(self, *args):
                self.decoder.incremental_eval(False)
        return IncrementalInference(self)

    def incremental_eval(self, mode=True):
        """Sets the decoder and all children in incremental evaluation mode."""
        assert self._is_incremental_eval != mode, \
            'incremental_eval already set to mode {}'.format(mode)

        self._is_incremental_eval = mode
        if mode:
            self.clear_incremental_state()

        def apply_incremental_eval(module):
            if module != self and hasattr(module, 'incremental_eval'):
                module.incremental_eval(mode)
        self.apply(apply_incremental_eval)

    def get_incremental_state(self, key):
        """Return cached state or None if not in incremental inference mode."""
        if self._is_incremental_eval and key in self._incremental_state:
            return self._incremental_state[key]
        return None

    def set_incremental_state(self, key, value):
        """Cache state needed for incremental inference mode."""
        if self._is_incremental_eval:
            self._incremental_state[key] = value
        return value

    def clear_incremental_state(self):
        """Clear all state used for incremental generation.

        **For incremental inference only**

        This should be called before generating a fresh sequence.
        beam_size is required if using BeamableMM.
        """
        if self._is_incremental_eval:
            self._incremental_state = {}

            def apply_clear_incremental_state(module):
                if module != self and hasattr(module, 'clear_incremental_state'):
                    module.clear_incremental_state()
            self.apply(apply_clear_incremental_state)

    def reorder_incremental_state(self, new_order):
        """Reorder buffered internal state (for incremental generation).

        **For incremental inference only**

        This should be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the choice of beams.
        """
        if self._is_incremental_eval:
            def apply_reorder_incremental_state(module):
                if module != self and hasattr(module, 'reorder_incremental_state'):
                    module.reorder_incremental_state(new_order)
            self.apply(apply_reorder_incremental_state)

    def set_beam_size(self, beam_size):
        """Sets the beam size in the decoder and all children."""
        def apply_set_beam_size(module):
            if module != self and hasattr(module, 'set_beam_size'):
                module.set_beam_size(beam_size)
        self.apply(apply_set_beam_size)
