import torch


class DataPrefetcher(object):

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.next_data = None
        self.next_target = None
        self.preload()

    def _load_data(self):
        try:
            self.next_data, self.next_target = next(self.loader)
        except StopIteration:
            self.next_data = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data = self.next_data.to('cuda').requires_grad_(False)
            self.next_target = self.next_target.to('cuda', non_blocking=True).requires_grad_(False)

    def preload(self):
        self._load_data()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        target = self.next_target
        if data is not None:
            data.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return data, target
