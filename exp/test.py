
N = 4
import math
sqr_size = math.floor(math.sqrt(N))
if math.sqrt(N).is_integer():
    p_b = p_t = p_r = p_l = 0
    col = row = sqr_size
else:
    remain_att = N - sqr_size**2
    if remain_att > sqr_size:
        p_b = p_r = 0
        p_l = p_t = 1
        col = row = sqr_size + 1
    else:
        p_b = 0
        p_t = p_r = p_l = 1
        col = sqr_size
        row = col + 1



print(row, col)
print(p_b, p_r, p_t, p_l)