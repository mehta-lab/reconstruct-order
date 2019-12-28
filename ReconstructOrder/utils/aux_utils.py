import os
import functools

def loop_pt(func):
    @functools.wraps(func)
    def wrapper_loop_pt(*args, **kwargs):
        img_io = kwargs['img_io']
        for pos_idx, pos_name in enumerate(img_io.pos_list):
            img_io.img_in_pos_path = os.path.join(img_io.img_sm_path, pos_name)
            img_io.pos_idx = pos_idx
            for t_idx in img_io.t_list:
                img_io.t_idx = t_idx
                kwargs['img_io'] = img_io
                func(*args, **kwargs)
    return wrapper_loop_pt

