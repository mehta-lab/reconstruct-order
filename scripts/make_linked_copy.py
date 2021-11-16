import os

def main():
    # input_dir = '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/SMS_2018_1227_1433_1_SMS_2018_1227_1433_1_registered_submit'
    # output_dir = '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/MBL_DL_image_translation/data'
    input_dir = '/CompMicro/projects/HEK/2021_07_29_LiveHEK_NoPerf_63x_09NA_tif/2021_07_29_LiveHEK_NoPerf_63x_09NA_tif_registered/Mock/HEK_63X_16_256_do20_zoom_0.8_1.2_noise_0.4_jitter_0.8_blur_0.1_10'
    output_dir = '/CompMicro/projects/HEK/2021_07_29_LiveHEK_NoPerf_63x_09NA_tif/2021_07_29_LiveHEK_NoPerf_63x_09NA_tif_registered/Mock_seg'
    # pos_ids = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    pos_ids = list(range(16))
    # z_ids = list(range(10, 16))
    z_ids = list(range(31, 36))  # HEK 63X
    channels = ['Retardance', 'phase', '405', '568']
    for fname in os.listdir(input_dir):
        if 'Orientation' in fname or 'Brightfield_computed' in fname:
            continue
        if 'im' in fname:
            # chan = fname.split("_")[1]
            print(fname)
            # pos_id = int(fname.split("_")[3].strip('p'))
            # z_id = int(fname.split("_")[4].strip('z').strip('.tif'))
            pos_id = int(fname.split("_")[2].strip('p'))
            z_id = int(fname.split("_")[3].strip('z'))
            # if chan in channels and pos_id in pos_ids and z_id in z_ids:
            if z_id in z_ids:
                os.link(os.path.join(input_dir, fname),
                        os.path.join(output_dir, fname))

if __name__ == '__main__':
    main()
