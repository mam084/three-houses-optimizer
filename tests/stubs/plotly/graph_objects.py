class Figure:
    def __init__(self):
        self.traces = []
        self.layout = {}

    def add_trace(self, trace):
        self.traces.append(trace)

    def update_layout(self, **kwargs):
        self.layout.update(kwargs)


class Scatterpolar:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class Bar:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
