import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from modules.tdmm_estimator import TDMMEstimator
from modules.flame_config import cfg as flame_cfg

from logger import Visualizer

from animate import normalize_kp
from animation_demo import find_best_frame


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    tdmm = TDMMEstimator(flame_cfg)

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
        generator.cuda()
        kp_detector.cuda()
        tdmm.cuda()
 
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    tdmm.load_state_dict(checkpoint['tdmm'])

    generator.eval()
    kp_detector.eval()
    tdmm.eval()

    return generator, kp_detector, tdmm


def make_animation(source_image,
                   driving_video,
                   driving_coefs,
                   generator, kp_detector, tdmm, with_eye=False,
                   relative=True, adapt_movement_scale=True, cpu=False,
                   forward_mask=0, backward_mask=0, alpha=1,
                   ):
    alpha = float(alpha)

    def batch_orth_proj(X, camera):
        camera = camera.clone().view(-1, 1, 3)
        X_trans = X[:, :, :2] + camera[:, :, 1:]
        X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
        shape = X_trans.shape
        Xn = (camera[:, :, 0:1] * X_trans)
        return Xn

    with torch.no_grad():
        predictions = []
        visualizations = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        kp_source = kp_detector(source)
        source_codedict = tdmm.encode(source)
        source_verts, source_transformed_verts, _ = tdmm.decode_flame(source_codedict)
        source_albedo = tdmm.extract_texture(source, source_transformed_verts, with_eye=with_eye)

        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        driving_initial = driving[:, :, 0].cuda()
        kp_driving_initial = kp_detector(driving[:, :, 0].cuda())
        driving_init_codedict = tdmm.encode(driving_initial)
        driving_init_codedict['shape'] = torch.tensor([driving_coefs[0]['shape'][:100].tolist()]).cuda()
        driving_init_codedict['exp'] = torch.tensor([driving_coefs[0]['exp'][:50].tolist()]).cuda()
        driving_init_codedict['pose'] = torch.tensor([driving_coefs[0]['pose'][:6].tolist()]).cuda()
        driving_init_verts, driving_init_transformed_verts, _ = tdmm.decode_flame(driving_init_codedict)

        for frame_idx in tqdm(range(len(driving_coefs))):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()

            kp_driving = kp_detector(driving_frame)
            driving_codedict = tdmm.encode(driving_frame)
            driving_codedict['shape'] = torch.tensor([driving_coefs[frame_idx]['shape'][:100].tolist()]).cuda()
            driving_codedict['exp'] = torch.tensor([driving_coefs[frame_idx]['exp'][:50].tolist()]).cuda()
            driving_codedict['pose'] = torch.tensor([driving_coefs[frame_idx]['pose'][:6].tolist()]).cuda() 

            # calculate relative 3D motion in the code space
            if relative:
                delta_shape = source_codedict['shape'] + alpha * (driving_codedict['shape'] - driving_init_codedict['shape'])
                delta_exp = source_codedict['exp'] + alpha * (driving_codedict['exp'] - driving_init_codedict['exp'])
                delta_pose = source_codedict['pose'] + alpha * (driving_codedict['pose'] - driving_init_codedict['pose'])
            else:
                delta_shape = source_codedict['shape']
                delta_exp = driving_codedict['exp']
                delta_pose = driving_codedict['pose']

            delta_source_verts, _, _ = tdmm.flame(shape_params=delta_shape,
                                           expression_params=delta_exp,
                                           pose_params=delta_pose)

            if relative:
                delta_scale = source_codedict['cam'][:, 0:1] * driving_codedict['cam'][:, 0:1] / driving_init_codedict['cam'][:, 0:1]
                delta_trans = source_codedict['cam'][:, 1:] + driving_codedict['cam'][:, 1:] - driving_init_codedict['cam'][:, 1:]
            else:
                delta_scale = driving_codedict['cam'][:, 0:1]
                delta_trans = driving_codedict['cam'][:, 1:]

            delta_cam = torch.cat([delta_scale, delta_trans], dim=1)
            delta_source_transformed_verts = batch_orth_proj(delta_source_verts, delta_cam)
            delta_source_transformed_verts[:, :, 1:] = - delta_source_transformed_verts[:, :, 1:]

            render_ops = tdmm.render(source_transformed_verts, delta_source_transformed_verts, source_albedo)

            # calculate relative kp
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            
            # remove effect of keypoints
            forward_mask = int(forward_mask)
            backward_mask = int(backward_mask)
            kp_norm['value'][:forward_mask] = kp_source['value'][:forward_mask].clone()
            kp_norm['jacobian'][:forward_mask] = kp_source['jacobian'][:forward_mask].clone()
            kp_norm['value'][-backward_mask:] = kp_source['value'][-backward_mask:].clone()
            kp_norm['jacobian'][-backward_mask:] = kp_source['jacobian'][-backward_mask:].clone()

            out = generator(source, kp_source=kp_source, kp_driving=kp_norm, render_ops=render_ops,
                                        driving_features=driving_codedict)
            del out['sparse_deformed']
            out['kp_source'] = kp_source
            out['kp_driving'] = kp_driving
            visualization = Visualizer(kp_size=5, draw_border=True, colormap='gist_rainbow').visualize(source=source,
                                                                                driving=driving_frame, out=out)
            visualizations.append(visualization)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

    return predictions, visualizations


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")

    parser.add_argument("--source_image_pth", default='', help="path to source image")
    parser.add_argument("--driving_video_pth", default='', help="path to driving video")
    parser.add_argument("--driving_coefs_pth", default='', help="path to driving coefs")
    parser.add_argument("--driving_audio_pth", default='', help="path to driving audio")
    parser.add_argument("--result_video_pth", default='./output/result.mp4', help="path to output")
    parser.add_argument("--result_vis_video_pth", default='./output/result_vis.mp4', help="path to output vis")
    parser.add_argument("--fps", default=25, help="frame per second")
 
    parser.add_argument("--with_eye", action="store_true", help="use eye part for extracting texture")
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")
    parser.add_argument("--best_frame", default=None)
 
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    parser.add_argument("-f", "--forward_mask", default=0)
    parser.add_argument("-b", "--backward_mask", default=0)
    parser.add_argument("-a", "--alpha", default=1)
 
    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    source_image = imageio.imread(opt.source_image_pth)
    reader = imageio.get_reader(opt.driving_video_pth)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    driving_coefs = np.load(opt.driving_coefs_pth, allow_pickle=True)

    if len(driving_video) > len(driving_coefs):
        driving_video = driving_video[:len(driving_coefs)]
    elif len(driving_video) < len(driving_coefs):
        driving_video = driving_video + driving_video[::-1] + driving_video
        driving_video = driving_video[:len(driving_coefs)]

    generator, kp_detector, tdmm = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

    i = int(opt.best_frame) if opt.best_frame else find_best_frame(source_image, driving_video, cpu=opt.cpu)
    print ("Best frame: " + str(i))
    driving_forward = driving_video[i:]
    driving_backward = driving_video[:(i+1)][::-1]
    driving_coefs_forward = driving_coefs[i:]
    driving_coefs_backward = driving_coefs[:(i+1)][::-1]
    predictions_forward, visualizations_forward = make_animation(
        source_image,
        driving_forward, 
        driving_coefs_forward,
        generator, kp_detector, tdmm,
        with_eye = opt.with_eye,
        relative = opt.relative,
        adapt_movement_scale = opt.adapt_scale,
        forward_mask = opt.forward_mask,
        backward_mask = opt.backward_mask,
        alpha = opt.alpha,
    )
    predictions_backward, visualizations_backward = make_animation(
        source_image,
        driving_backward, 
        driving_coefs_backward,
        generator, kp_detector, tdmm,
        with_eye = opt.with_eye,
        relative = opt.relative,
        adapt_movement_scale = opt.adapt_scale,
        cpu = opt.cpu,
        forward_mask = opt.forward_mask,
        backward_mask = opt.backward_mask,
        alpha = opt.alpha,
    )
    predictions = predictions_backward[::-1] + predictions_forward[1:]
    visualizations = visualizations_backward[::-1] + visualizations_forward[1:]
        
    imageio.mimsave(opt.result_video_pth, [img_as_ubyte(frame) for frame in predictions], fps=fps)
    imageio.mimsave(opt.result_vis_video_pth, visualizations)
    
    final_result_pth = opt.result_video_pth.replace("result", "result_with_audio")
    command = f"ffmpeg -y -i {opt.result_video_pth} -i {opt.driving_audio_pth} -map 0:v -map 1:a {final_result_pth}"
    os.system(command)
