from torch.optim.lr_scheduler import ReduceLROnPlateau


class PlateauReducer(ReduceLROnPlateau):
    """
    A wrapper for torch.optim.lr_scheduler.ReduceLROnPlateau.
    The only difference is that it tracks the metric specified on init.
    """

    def __init__(self, track_metric, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = track_metric

    def step(self):
        super().step(self.metric.value)
