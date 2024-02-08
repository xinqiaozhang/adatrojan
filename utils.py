import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getQScore(ten):
    ten_tensor = torch.tensor(ten, dtype=torch.float32).to(device)
    avgTen = torch.mean(ten_tensor, dim=0)
    sortedTen, _ = torch.sort(avgTen)
    Q = (abs(sortedTen[-1] - sortedTen[-2]))/(sortedTen[-1] - sortedTen[0])
    return Q