
import torch

class IncrCorr:
    """Calculate correlation between two sets of sequences, incrementally

    An object of this class calculates and stores the running correlation of a set of
    sequences (referred to as "u" and "v").

    The sequences can be tensors of any dimension (provided that u and v are the same
    size). The correlation is calculated between u and v along an axis of interest
    (the "axis").

    The tensors u and v are provided in pieces by repeatedly calling "update;"
    the final calculation is as if each u and v were concatenated with each previous u
    and v, respectively, along the axis.

    The result is the same dimension as u and v, but with the axis collapsed to a size of
    one (since the correlation results in a single number relating all examples along
    that axis in isolation).

    This approach is based on Numerically Stable, Single-Pass, Parallel Statistics
    Algorithms" by Bennett et. al. (2009). Equation numbers as well as variable naming
    below refer to that paper.

    Note that the updates need not be fixed length, though for every update, u and v
    must be the same size. For example, if the axis is 1, and we wish to provide two
    updates, u and v could be of size (4,6,7) for the first update and (4,9,7) for the
    second. The final correlation would be calculated over a tensor of size (4,15,7)
    -- i.e. the first update concatenated with the second -- and would store individual
    correlation results in a tensor of size (4,1,7).
    """

    @staticmethod
    def _contract_dim(dims, axis):
        """Get dimensions of the final correlations, given an example input"""
        cdims = list(dims)
        cdims[axis] = 1
        return cdims

    def __init__(self, dims, axis):
        """Initialization

        dims: dimensions of tensors to be used to calculation correlation
        axis: axis along which to perform the calculation

        Note that the dimension along the specified axis can change for
        each update batch, so this dimension (within dims) can be anything for the
        purposes of init.
        """
        if not isinstance(dims, tuple):
            raise TypeError("dim must be a tuple of integers")
        if not isinstance(axis, int):
            raise TypeError("axis must be an int")
        if axis >= len(dims) or axis < 0:
            raise ValueError("You must have 0 <= axis < {}".format(len(dims)))

        self.axis = axis
        self.dims = self._contract_dim(dims, axis)

        self.reset()


    def _full_mom(self, M2_L1, delt_21, x2, n1, n2, n, mu_2):
        """Find new running moment including new samples

        delt_21: the difference in means between previous samples and new ones
        x2: the new samples
        n1: the number of previous samples (along desired axis)
        n2: the number of new samples (along desired axis)
        n: the total number of samples, including new ones

        Return: The moment of whole data set for u or v sequences
        """

        # Calculate the moment of just the new test set
        M2_L2 = torch.pow(x2 - mu_2, 2).sum(dim=self.axis, keepdim=True)

        # Calculate the moment itself according to II.4
        M2_L = M2_L1 + M2_L2 + ((n1*n2)/n) * torch.pow(delt_21, 2)

        return M2_L

    def _full_cov(self, n, n2, x2_u, x2_v, mu_u2, mu_v2):
        """Find new running covariance including new samples

        n: total number of samples, including new ones
        n2: number of samples in new dataset (along the desired axis)
        x2_u: new u sequences
        x2_v: new v sequences
        mu_u2: means of new u sequences
        mu_v2: means of new v sequences

        Return: the covariance of the whole data set, including the new samples
        """

        # Calculate covariance for new sequences in isolation
        # Dot product of corresponding values in each sequence
        # along the axis specified
        C2_L2 = torch.sum((x2_u - mu_u2) * (x2_v - mu_v2), dim=self.axis, keepdim=True)

        # Calculate the full covariance itself according to III.6
        delt_u21 = mu_u2 - self.mu_u1
        delt_v21 = mu_v2 - self.mu_v1
        C2_L = self.C2_L1 + C2_L2 + ((self.n1 * n2)/n) * delt_u21 * delt_v21

        return C2_L

    def update(self, new_u_seq, new_v_seq):
        """Update the running correlation with new examples

        new_u_seq: tensor, with dimensions dim as specified in init
        new_v_seq: tensor, with same dimensions
        """
        if new_u_seq.device != self.C2_L1.device:
            self.C2_L1 = self.C2_L1.to(new_u_seq.device)
            self.M2_L1_u = self.M2_L1_u.to(new_u_seq.device)
            self.M2_L1_v = self.M2_L1_v.to(new_u_seq.device)
            self.mu_u1 = self.mu_u1.to(new_u_seq.device)
            self.mu_v1 = self.mu_v1.to(new_u_seq.device)

        x2_u = new_u_seq
        x2_v = new_v_seq

        for inp in (x2_u, x2_v):
            if not isinstance(inp, torch.Tensor):
                raise TypeError("both inputs to update must be tensors")
            if self._contract_dim(x2_u.shape, self.axis) != self.dims:
                msg = "both inputs to update must have shape {}"
                raise ValueError(msg.format(self.dims))

        # Calculate number of samples
        n2 = new_u_seq.shape[self.axis]
        n = self.n1 + n2

        # Calculate means of new sequences
        mu_u2 = torch.mean(new_u_seq, self.axis, keepdim=True)
        mu_v2 = torch.mean(new_v_seq, self.axis, keepdim=True)

        # Find the difference in means between previous samples and current ones
        delt_u21 = mu_u2 - self.mu_u1
        delt_v21 = mu_v2 - self.mu_v1

        # Get the full mean according to II.3
        mu_u = self.mu_u1 + n2 * delt_u21 / n
        mu_v = self.mu_v1 + n2 * delt_v21 / n

        # Get full moments
        M2_L_u = self._full_mom(self.M2_L1_u, delt_u21, x2_u, self.n1, n2, n, mu_u2)
        M2_L_v = self._full_mom(self.M2_L1_v, delt_v21, x2_v, self.n1, n2, n, mu_v2)

        # Get full covariance
        C2_L = self._full_cov(n, n2, x2_u, x2_v, mu_u2, mu_v2)

        # Update internal variables with new values
        self.C2_L1 = C2_L
        self.M2_L1_u = M2_L_u
        self.M2_L1_v = M2_L_v
        self.mu_u1 = mu_u
        self.mu_v1 = mu_v
        self.n1 = n

    def retrieve(self):
        """Get the current running correlation"""
        return self.C2_L1 / torch.sqrt(self.M2_L1_u * self.M2_L1_v)

    def reset(self):
        """Reset the internal state to start new calculations"""

        self.C2_L1 = torch.zeros(self.dims) # Running correlations
        self.M2_L1_u = torch.zeros(self.dims) # Running moment for u sequences
        self.M2_L1_v = torch.zeros(self.dims) # Running moment for v sequences
        self.mu_u1 = torch.zeros(self.dims) # Running means for u
        self.mu_v1 = torch.zeros(self.dims) # Running means for v
        self.n1 = 0 # Total number of examples seen
