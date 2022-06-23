import torch
import torchmetrics


class CpcAccuracy(torchmetrics.Metric):
    def __init__(self, loss):
        super().__init__(compute_on_step=False)

        self.loss = loss
        self.add_state('batch_results', default=torch.tensor([0.0]))
        self.add_state('total_count', default=torch.tensor([0]))

    def update(self, *input):
        self.batch_results += self.loss.cpc_accuracy(*input)
        self.total_count += 1

    def compute(self):
        return self.batch_results / self.total_count.float()