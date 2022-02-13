
def count_params(model):
    cpt = 0
    for x in model.parameters():
        cpt += x.numel()
    return cpt
