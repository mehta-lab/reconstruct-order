import os
import functools

def loop_pt(func):
    @functools.wraps(func)
    def wrapper_loop_pt(*args, **kwargs):
        img_io = kwargs['img_io']
        pos_idx = kwargs['pos_idx']
        for pos in range(len(img_io.size_p)):
            pos_idx = pos
            for t_idx in img_io.t_list:
                img_io.t_idx = t_idx
                kwargs['img_io'] = img_io
                func(*args, **kwargs)
    return wrapper_loop_pt

