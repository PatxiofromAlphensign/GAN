import numpy as np

def make2Dndarray(x, out=[], final=[]):
    if len(x.shape) < 1:
        return final
    for e in x:
        if len(e.shape) > 0:
            for e1 in e:
                out.append(e1)
        else:
            out.append(e) 
    final = np.array(out)
    return make2Dndarray(x[1], out=out, final=final)


