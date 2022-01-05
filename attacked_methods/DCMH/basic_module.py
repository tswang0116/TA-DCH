import torch as t
import time


class BasicModule(t.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(t.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(t.load(path))

    def save(self, name=None):
        if name is None:
            prefix = self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), 'checkpoints/' + name)
        return name

    def forward(self, *input):
        pass

