import torch

class FGBGLoader(object):

    def __init__(self, dataset, batch_size=16):
        self.dataset    = dataset
        self.batch_size = batch_size

        # number of subprocesses to use for data loading
        self.num_workers = 0

        seed = 1
        # For reproducibility
        torch.manual_seed(seed)

        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda, 'Batch_size:', self.batch_size))
        

        if cuda:
            self.batch_size = batch_size
            self.num_workers = 4
            self.pin_memory = True
        else:
            self.shuffle = True
            self.batch_size = batch_size

    def __call__(self):
        return torch.utils.data.DataLoader(dataset=self.traindataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, shuffle=True, pin_memory=True)