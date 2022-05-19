def vcol(oneDarray):
    return oneDarray.reshape((oneDarray.size, 1))
def vrow(oneDarray):
    return oneDarray.reshape((1, oneDarray.size))