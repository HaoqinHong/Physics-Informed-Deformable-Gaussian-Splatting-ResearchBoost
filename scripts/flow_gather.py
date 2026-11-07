"""Standalone exact copy of the helper applied by patches/pidg-flow.patch."""
import torch

class _ZeroElidingGather(torch.autograd.Function):
    """Upstream row gather with ordered dense accumulation of nonzero terms."""

    @staticmethod
    def forward(ctx, table, indices):
        ctx.save_for_backward(indices)
        ctx.table_shape = table.shape
        return table[indices]

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        rows = indices.reshape(-1)
        values = grad_output.reshape(rows.numel(), *ctx.table_shape[1:])
        result = grad_output.new_zeros(ctx.table_shape)
        # With create_graph, keep every addend so that higher derivatives at
        # zero incoming gradients retain the upstream linear map as well.
        if not torch.is_grad_enabled() and values.numel():
            active = values.reshape(rows.numel(), -1).ne(0).any(dim=1).nonzero().flatten()
            rows = rows.index_select(0, active)
            values = values.index_select(0, active)
        # nonzero() preserves row order. Only exact zero vectors were omitted;
        # nonfinite values and every nonzero addend remain in the dense sum.
        rows = torch.where(rows < 0, rows + ctx.table_shape[0], rows)
        result.index_put_((rows,), values, accumulate=True)
        return result, None


def _flow_gather(table, indices):
    return _ZeroElidingGather.apply(table, indices)
