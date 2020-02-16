from typing import Callable

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AvgMetric:
    """Represents a simple average of a metric accumulated over multiple steps."""

    def __init__(
        self,
        fn: Callable[..., torch.FloatTensor],
        init_val: float = 0.0,
        device: torch.device = DEVICE,
        name: str = None,
    ):
        self._fn = fn
        self._init_val = torch.Tensor(init_val).to(device).detach()
        self._val = self._init_val.clone().detach()
        self._steps = 0
        self.name = name

    def accumulate(self, **kwargs) -> torch.Tensor:
        val = self._fn(**kwargs)
        self._val += val
        self._steps += 1
        return val

    def compute(self) -> torch.Tensor:
        return self._val / self._steps if self._steps else self._val

    def reset(self) -> None:
        self._val = self._init_val.clone().detach()
        self._steps = 0

    def compute_and_reset(self) -> torch.Tensor:
        accum_val = self.compute()
        self.reset()
        return accum_val


def selective_loss_fn(
    targets: torch.Tensor,
    f_out: torch.Tensor,
    g_out: torch.Tensor,
    target_coverage: float,
    lambd: int = 32,
) -> torch.Tensor:
    """
    Calculates the selective loss.

    Args:
        targets: .
        preds: .
        c: .
        lambd: .

    Returns:
        .
    """

    def emp_sr(
        targets: torch.Tensor, f_out: torch.Tensor, g_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates empircal selective risk, as defined in equiation (2) of the
        SelectiveNet paper.

        TODO: why do they not normalize this by the emprical coverage?

        Args:
            targets: .
            f_out: .
            g_out: .

        Returns:
            .
        """

        return (
            torch.nn.functional.cross_entropy(
                f_out, torch.argmax(targets, dim=1), reduction="none"
            )
            * g_out.squeeze()
        ).mean()

    def emp_cov(g_out: torch.Tensor) -> torch.Tensor:
        """Calculates empircal coverage."""
        return torch.mean(g_out)

    def psi(a: torch.Tensor) -> torch.Tensor:
        """Quadratic penalty function."""
        return torch.pow(torch.max(torch.zeros_like(a), a), 2)

    return emp_sr(targets, f_out, g_out) + lambd * psi(target_coverage - emp_cov(g_out))


def selective_acc_fn(targets, f_out, g_out, threshold=0.5):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, float) -> torch.Tensor
    selected = torch.squeeze(torch.gt(g_out, threshold))
    correct = torch.eq(torch.argmax(f_out, dim=1), torch.argmax(targets, dim=1))
    num_selected = torch.sum(selected)
    num_selected_and_correct = torch.sum(selected & correct)
    return 1.0 * num_selected_and_correct / num_selected


def rejected_acc_fn(
    targets: torch.Tensor,
    f_out: torch.Tensor,
    g_out: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    rejected = torch.squeeze(torch.le(g_out, threshold))
    correct = torch.eq(torch.argmax(f_out, dim=1), torch.argmax(targets, dim=1))
    num_rejected = torch.sum(rejected)
    num_rejected_and_correct = torch.sum(rejected & correct)
    return num_rejected_and_correct / num_rejected


def acc_fn(targets: torch.Tensor, f_out: torch.Tensor) -> torch.Tensor:
    correct = torch.eq(torch.argmax(f_out, dim=1), torch.argmax(targets, dim=1))
    return torch.sum(correct) / len(f_out)


def coverage_fn(
    targets: torch.Tensor,
    f_out: torch.Tensor,
    g_out: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    return torch.mean(torch.squeeze(torch.gt(g_out, threshold)).float())


def loss_fn(
    targets: torch.Tensor,
    f_out: torch.Tensor,
    g_out: torch.Tensor,
    h_out: torch.Tensor,
    alpha: float,
    target_coverage: float,
    lambd=32,
):
    loss_f_g = selective_loss_fn(targets, f_out, g_out, target_coverage, lambd=lambd)
    loss_h = torch.nn.functional.cross_entropy(h_out, torch.argmax(targets, dim=1))
    loss_all = alpha * loss_f_g + (1 - alpha) * loss_h
    return loss_all
