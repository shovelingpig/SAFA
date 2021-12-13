import matplotlib
matplotlib.use('Agg')
import os
from argparse import ArgumentParser

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte

from stv_demo import load_checkpoints, make_animation


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-v", "--version", required=True)
    opt = parser.parse_args()

    input_dir = "./test/assets"
    output_dir = f"./test/results/version_{opt.version}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    generator, kp_detector, tdmm = load_checkpoints(config_path="./config/end2end.yaml", checkpoint_path="./ckpt/final_3DV.tar")

    paths = os.listdir(input_dir)
    for path in paths:
        data_dir = os.path.join(input_dir, path)
        source_image_pth = os.path.join(data_dir, "image.npy")
        driving_video_pth = os.path.join(data_dir, "video.mp4")
        driving_coefs_pth = os.path.join(data_dir, "flame_coefs.npy")
        driving_audio_pth = os.path.join(data_dir, "video.mp4")

        result_dir = os.path.join(output_dir, path)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        result_video_pth = os.path.join(result_dir, "result.mp4")
        result_vis_video_pth = os.path.join(result_dir, "result_vis.mp4")
        result_with_audio_pth = os.path.join(result_dir, "result_with_audio.mp4")

        # load image
        source_image = np.load(source_image_pth)
        source_image = resize(source_image, (256, 256))[..., :3]

        # load video
        reader = imageio.get_reader(driving_video_pth)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

        # load coefs
        driving_coefs = np.load(driving_coefs_pth, allow_pickle=True)

        # inference
        predictions, visualizations = make_animation(
            source_image,
            driving_video,
            driving_coefs,
            generator, kp_detector, tdmm,
        )

        # save result
        imageio.mimsave(result_video_pth, [img_as_ubyte(frame) for frame in predictions], fps=fps)
        imageio.mimsave(result_vis_video_pth, visualizations)
        command = f"ffmpeg -y -i {result_video_pth} -i {driving_audio_pth} -map 0:v -map 1:a {result_with_audio_pth}"
        os.system(command)
