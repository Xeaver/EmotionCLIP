import math
import warnings
from typing import Optional, Union, Any

from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau


class LRWarmupScheduler:
    def __init__(
        self,
        wrapped_scheduler: _LRScheduler,
        by_epoch: bool = True,
        epoch_len: Optional[int] = None,
        warmup_t: int = 0,
        warmup_by_epoch: bool = False,
        warmup_mode: str = "fix",
        warmup_init_lr: Optional[float] = None,
        warmup_init_factor: Optional[float] = None,
    ):
        # wrapped pytorch scheduler which is used for after warmup stage
        self.wrapped_scheduler = wrapped_scheduler
        # wrapped_scheduler updates by epoch or by iter
        self.by_epoch = by_epoch
        # the length of each epoch, only required when by_epoch=True and warmup_by_epoch=False
        self.epoch_len = epoch_len
        # warmup_epochs when warmup_by_epoch=True
        # warmup_iters when warmup_by_epoch=False
        self.warmup_t = warmup_t
        # warmup updates by epoch or by iter
        self.warmup_by_epoch = warmup_by_epoch
        # 'fix' (TIMM), 'auto' (Detector2), 'factor' (MMCV)
        self.warmup_mode = warmup_mode
        # initial lr = warmup_init_lr when mode='fix'
        self.warmup_init_lr = warmup_init_lr
        # initial lr = base_lr * warmup_init_factor, when mode='auto'/'factor'
        self.warmup_init_factor = warmup_init_factor

        if warmup_by_epoch:
            assert by_epoch
        if by_epoch and warmup_t and not warmup_by_epoch:
            assert epoch_len is not None
        if self._is_plateau:
            assert by_epoch

        self.param_groups = self.wrapped_scheduler.optimizer.param_groups
        self.base_lrs = [param_group["lr"] for param_group in self.param_groups]

        if warmup_t:
            # pre-compute the regular lr if no warmup is performed
            max_t = warmup_t // epoch_len if by_epoch and not warmup_by_epoch else warmup_t
            self.regular_lrs_per_t = self._pre_compute_regular_lrs_per_t(max_t)

        self.last_iter = self.last_epoch = 0
        self.in_iter_warmup = False


        if warmup_t > 0:
            if warmup_mode == "fix":
                assert isinstance(warmup_init_lr, float)
                self._set_lrs(warmup_init_lr)
            elif warmup_mode == "factor":
                assert isinstance(warmup_init_factor, float)
                self._set_lrs([base_lr * warmup_init_factor for base_lr in self.base_lrs])
            elif warmup_mode == "auto":
                assert isinstance(warmup_init_factor, float)
                self.warmup_end_lrs = self.regular_lrs_per_t[-1]
                self._set_lrs([base_lr * warmup_init_factor for base_lr in self.base_lrs])
            else:
                raise ValueError(f"Invalid warmup mode: {warmup_mode}")


    @property
    def _is_plateau(self) -> bool:
        return isinstance(self.wrapped_scheduler, ReduceLROnPlateau)


    def _pre_compute_regular_lrs_per_t(self, max_t: int) -> list[list[float]]:
        regular_lrs_per_t = [self.base_lrs]
        if self._is_plateau:
            return regular_lrs_per_t * (max_t + 1)
        for _ in range(max_t):
            self.wrapped_scheduler.step()
            regular_lrs_per_t.append([param_group["lr"] for param_group in self.param_groups])
        return regular_lrs_per_t


    def _get_warmup_lrs(self, t: int, regular_lrs: list[float]) -> list[float]:
        alpha = t / self.warmup_t
        if self.warmup_mode == "fix":
            return [
                self.warmup_init_lr * (1 - alpha) + base_lr * alpha for base_lr in self.base_lrs
            ]
        elif self.warmup_mode == "factor":
            factor = self.warmup_init_factor * (1 - alpha) + alpha
            return [lr * factor for lr in regular_lrs]
        elif self.warmup_mode == 'auto':
            return [
                base_lr * self.warmup_init_factor * (1 - alpha) + end_lr * alpha
                for base_lr, end_lr in zip(self.base_lrs, self.warmup_end_lrs)
            ]


    def _set_lrs(self, lrs: Union[float, list[float]]) -> None:
        if not isinstance(lrs, (list, tuple)):
            lrs = [lrs] * len(self.param_groups)
        for param_group, lr in zip(self.param_groups, lrs):
            param_group["lr"] = lr


    def epoch_step(self, metric: Optional[float] = None) -> None:
        if not self.by_epoch:
            return
        
        self.last_epoch += 1
        if self.warmup_by_epoch and self.last_epoch < self.warmup_t:
            self._set_lrs(
                self._get_warmup_lrs(self.last_epoch, self.regular_lrs_per_t[self.last_epoch]))
        elif self.warmup_by_epoch and self.last_epoch == self.warmup_t:
            self._set_lrs(self.regular_lrs_per_t[-1])
        elif not self.in_iter_warmup:
            if self._is_plateau:
                self.wrapped_scheduler.step(metric)
            else:
                self.wrapped_scheduler.step()


    def iter_step(self) -> None:
        if self.warmup_by_epoch:
            return

        self.last_iter += 1
        if self.last_iter < self.warmup_t:
            self.in_iter_warmup = True
            t = self.last_iter // self.epoch_len if self.by_epoch else self.last_iter
            self._set_lrs(self._get_warmup_lrs(self.last_iter, self.regular_lrs_per_t[t]))
        elif self.last_iter == self.warmup_t:
            self._set_lrs(self.regular_lrs_per_t[-1])
        else:
            self.in_iter_warmup = False
            if not self.by_epoch:
                self.wrapped_scheduler.step()


    def state_dict(self) -> dict[str, Any]:
        state = {key: value for key, value in self.__dict__.items() if key != "wrapped_scheduler"}
        state["wrapped_scheduler"] = self.wrapped_scheduler.state_dict()
        return state


    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.wrapped_scheduler.load_state_dict(state_dict.pop("wrapped_scheduler"))
        self.__dict__.update(state_dict)