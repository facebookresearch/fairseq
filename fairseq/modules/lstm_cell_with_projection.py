"""
An LSTM with Recurrent Dropout, a hidden_state which is projected and
clipping on both the hidden state and the memory state of the LSTM.
"""
import math
import torch
import itertools
import torch.nn as nn
from torch.autograd import Variable

class LstmCellWithProjection(nn.Module):
    """
    An LSTM with Recurrent Dropout and
    a projected and clipped hidden state and memory.
    Note: this implementation is slower than the native Pytorch LSTM because
    it cannot make use of CUDNN optimizations for stacked RNNs due to and
    variational dropout and the custom nature of the cell state.
    Parameters
    ----------
    input_size : ``int``, required.
        The dimension of the inputs to the LSTM.
    hidden_size : ``int``, required.
        The dimension of the outputs of the LSTM.
    cell_size : ``int``, required.
        The dimension of the memory cell used for the LSTM.
    go_forward: ``bool``, optional (default = True)
        The direction in which the LSTM is applied to the sequence.
        Forwards by default, or backwards if False.
    recurrent_dropout_probability: ``float``, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in
        Recurrent Neural Networks <https://arxiv.org/abs/1512.05287>`_ .
        Implementation wise, this simply applies a fixed dropout mask
        per sequence to the recurrent connection of the LSTM.
    state_projection_clip_value: ``float``, optional, (default = None)
        The magnitude with which to clip the hidden_state after projecting it.
    memory_cell_clip_value: ``float``, optional, (default = None)
        The magnitude with which to clip the memory cell.
    Returns
    -------
    output_accumulator : ``torch.FloatTensor``
        The outputs of the LSTM for each timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    final_state: ``Tuple[torch.FloatTensor, torch.FloatTensor]``
        The final (state, memory) states of the LSTM, with shape
        (1, batch_size, hidden_size) and  (1, batch_size, cell_size)
        respectively. The first dimension is 1 in order to match the Pytorch
        API for returning stacked LSTM states.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 cell_size,
                 go_forward,
                 recurrent_dropout_probability=0.0,
                 memory_cell_clip_value=None,
                 state_projection_clip_value=None,
                 is_training=True):

        super(LstmCellWithProjection, self).__init__()

        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.training = is_training

        self.go_forward = go_forward
        self.state_projection_clip_value = state_projection_clip_value
        self.memory_cell_clip_value = memory_cell_clip_value
        self.recurrent_dropout_probability = recurrent_dropout_probability

        # We do the projections for all the gates all at once.
        self.input_linearity = nn.Linear(input_size, 4 * cell_size, bias=False)
        self.state_linearity = nn.Linear(hidden_size, 4 * cell_size, bias=True)

        # Additional projection matrix for making the hidden state smaller.
        self.state_projection = nn.Linear(cell_size, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # Use sensible default initializations for parameters.
        block_orthogonal(self.input_linearity.weight.data, [self.cell_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.cell_size, self.hidden_size])
        stdv = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.state_projection.weight.data, -stdv, stdv)
        self.state_linearity.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.state_linearity.bias.data[self.cell_size:2 * self.cell_size].fill_(1.0)

    def forward(self, inputs,
                initial_state=None):
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, num_timesteps, input_size)
            to apply the LSTM over.
        initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. The ``state`` has shape (1, batch_size, hidden_size) and the
            ``memory`` has shape (1, batch_size, cell_size).
        Returns
        -------
        output_accumulator : ``torch.FloatTensor``
            The outputs of the LSTM for each timestep. A tensor of shape
            (batch_size, max_timesteps, hidden_size) where for a given batch
            element, all outputs past the sequence length for that batch are
            zero tensors.
            Can be used as input to next layer.
        final_state : ``Tuple[``torch.FloatTensor, torch.FloatTensor]``
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. The ``state`` has shape (1, batch_size, hidden_size) and the
            ``memory`` has shape (1, batch_size, cell_size).
            the very last output of the lstm cell in that layer.
        """
        batch_size, total_timesteps, _ = inputs.shape
        output_accumulator = []

        if initial_state is None:
            previous_memory = Variable(inputs.data.new(batch_size, self.cell_size).fill_(0), requires_grad=True)
            previous_state = Variable(inputs.data.new(batch_size, self.hidden_size).fill_(0), requires_grad=True)
        else:
            previous_state = initial_state[0].squeeze(0)
            previous_memory = initial_state[1].squeeze(0)

        if self.recurrent_dropout_probability > 0.0 and self.training:
            dropout_mask = get_dropout_mask(self.recurrent_dropout_probability,
                                            previous_state)
        else:
            dropout_mask = None

        for timestep in range(total_timesteps):
            # The index depends on which end we start.
            if self.go_forward:
                index = timestep
            else:
                index = total_timesteps - timestep - 1

            projected_input = self.input_linearity(inputs[:,index])
            projected_state = self.state_linearity(previous_state)



            projected_input_ig, projected_input_fg, projected_input_mi, projected_input_og = torch.split(projected_input, self.cell_size, 1)
            projected_state_ig, projected_state_fg, projected_state_mi, projected_state_og = torch.split(projected_state, self.cell_size, 1)

            # Main LSTM equations using relevant chunks of the big linear
            # projections of the hidden state and inputs.

            input_gate = torch.sigmoid(projected_input_ig + projected_state_ig)
            forget_gate = torch.sigmoid(projected_input_fg + projected_state_fg)
            memory_init = torch.tanh(projected_input_mi + projected_state_mi)
            output_gate = torch.sigmoid(projected_input_og + projected_state_og)
            memory = input_gate * memory_init + forget_gate * previous_memory

            # Here is the non-standard part of this LSTM cell; first, we clip the
            # memory cell, then we project the output of the timestep to a smaller size
            # and again clip it.

            if self.memory_cell_clip_value:

                memory = torch.clamp(memory, -self.memory_cell_clip_value, self.memory_cell_clip_value)

            pre_projection_timestep_output = output_gate * torch.tanh(memory)

            timestep_output = self.state_projection(pre_projection_timestep_output)

            if self.state_projection_clip_value:

                timestep_output = torch.clamp(timestep_output,
                                              -self.state_projection_clip_value,
                                              self.state_projection_clip_value)

            # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
            if dropout_mask is not None:

                timestep_output = timestep_output * dropout_mask

            previous_memory = memory
            previous_state = timestep_output
            output_accumulator.append(timestep_output)

        final_state = (previous_state.unsqueeze(0),
                       previous_memory.unsqueeze(0))

        if not self.go_forward:
            output_accumulator = output_accumulator[::-1]

        output_accumulator = torch.transpose(torch.stack(output_accumulator), 0, 1)
     
        return output_accumulator, final_state


def get_dropout_mask(dropout_probability, tensor_for_masking):
    binary_mask = tensor_for_masking.clone()
    binary_mask.data.copy_(torch.rand(tensor_for_masking.size()) > dropout_probability)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask.type_as(tensor_for_masking)


def block_orthogonal(tensor, split_sizes, gain=1.0):
    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise Exception("tensor dimensions must be divisible by their respective"
                        "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split))
               for max_size, split in zip(sizes, split_sizes)]
    for block_start_indices in itertools.product(*indexes):

        index_and_step_tuples = zip(block_start_indices, split_sizes)

        block_slice = tuple([slice(start_index, start_index + step)
                             for start_index, step in index_and_step_tuples])
        tensor[block_slice] = torch.nn.init.orthogonal_(tensor[block_slice].contiguous(), gain=gain)
