import torch



if torch.cuda.is_available():
    device= torch.device("cuda")
    map_loc = 'cuda'
else:
    device= torch.device("cpu")
    map_loc = 'cpu'

device = torch.device("cpu")


map_loc = 'cpu'