import os
import numpy as np
from ReconstructOrder.utils.mManagerIO import mManagerReader

if __name__ == '__main__':

    input_chan = output_chan = ['Phase3D', 'Retardance']  # first channel is the reference channel
    input_dir = '/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/'
    output_dir = '/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_input'
    img_io = mManagerReader(input_dir, output_dir, input_chans=input_chan, output_chans=output_chan)
    img_io.pos_idx = 0
    img_io.t_idx = 0
    img_io.binning = 1
    z_ids = list(range(26, 41))

    if not os.path.exists(img_io.img_sm_path):
        raise FileNotFoundError(
            "image file doesn't exist at:", img_io.img_sm_path
        )
    os.makedirs(img_io.img_output_path, exist_ok=True)

    for pos_idx in range(img_io.n_pos):  # nXY
        img_io.pos_idx = pos_idx
        for z_idx in z_ids:
            img_io.z_idx = z_idx
            imgs_tc = []
            for t_idx in range(img_io.n_time):
                print('Processing position %03d, time %03d, z %03d...' % (pos_idx, t_idx, z_idx))
                img_io.t_idx = t_idx
                imgs_c = []
                for chan_idx in range(len(input_chan)):
                    img_io.chan_idx = chan_idx
                    img = img_io.read_img()
                    imgs_c.append(img)
                # dynamorph uses tyxc format
                imgs_c = np.stack(imgs_c, axis=-1)
                imgs_tc.append(imgs_c)
            imgs_tc = np.stack(imgs_tc, axis=0)
            imgs_tc = imgs_tc.astype(np.float32)
            img_name = 'img_t%03d_p%03d_z%03d.npy' % (t_idx, pos_idx, z_idx)
            np.save(os.path.join(output_dir, img_name), imgs_tc)