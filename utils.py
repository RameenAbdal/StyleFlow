import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import tensorflow as tf


import PIL.ImageFile

import scipy.ndimage

import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import os
import re
import sys

import pretrained_networks

def Align_face_image(src_file, output_size=1024, transform_size=4096,
                     enable_padding=True):
    print('aligning image...')
    import dlib
    img_ = dlib.load_rgb_image(src_file)
    print("Image Shape :", img_.shape)

    frontal_face = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")  # cnn model
    shape_ = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # same as ffhq dataset
    dets = frontal_face(img_, 1)
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(i, d.rect.left(),
                                                                                          d.rect.top(),
                                                                                          d.rect.right(),
                                                                                          d.rect.bottom(),
                                                                                          d.confidence))
        shape = shape_(img_, d.rect)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0).x, shape.part(1)))

        # Parse landmarks.
        # pylint: disable=unused-variable

        lm_chin = np.array([[shape.part(i).x, shape.part(i).y] for i in range(17)])
        lm_eyebrow_left = np.array([[shape.part(i).x, shape.part(i).y] for i in range(17, 22)])
        lm_eyebrow_right = np.array([[shape.part(i).x, shape.part(i).y] for i in range(22, 27)])
        lm_nose = np.array([[shape.part(i).x, shape.part(i).y] for i in range(27, 31)])
        lm_nostrils = np.array([[shape.part(i).x, shape.part(i).y] for i in range(31, 36)])
        lm_eye_left = np.array([[shape.part(i).x, shape.part(i).y] for i in range(36, 42)])
        lm_eye_right = np.array([[shape.part(i).x, shape.part(i).y] for i in range(42, 48)])
        lm_mouth_outer = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 60)])
        lm_mouth_inner = np.array([[shape.part(i).x, shape.part(i).y] for i in range(60, 68)])

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Load in-the-wild image.


        if not os.path.isfile(src_file):
            print('\nCannot find source image. Please run "--wilds" before "--align".')
            return
        img = PIL.Image.open(src_file)

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                              1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                            PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
        img.save(src_file)

def gram_matrix(input_tensor):
    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
    """Expects two images of dimension h, w, c"""
    # height, width, num filters of each laye
    base_style = tf.reshape(base_style, [base_style.shape[1], base_style.shape[2], base_style.shape[3]])
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target))




#----------------------------------------------------------------------------

def generate_im_official(network_pkl='gdrive:networks/stylegan2-ffhq-config-f.pkl', seeds=[22], truncation_psi=0.5):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('seed%04d.png' % seed))





def generate_im_from_random_seed(Gs, seed=22, truncation_psi=0.5):

    seeds = [seed]
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
        images = Gs.run(z, None, **Gs_kwargs)  # [minibatch, height, width, channel]
        # PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('seed%04d.png' % seed))
    return images



class Build_model:
    def __init__(self, opt):

        self.opt = opt
        if os.path.exists("/usr/app/stylegan/stylegan2-ffhq-config-f.pkl"):
            print("Found local StyleGan2 !")
            network_pkl = "/usr/app/stylegan/stylegan2-ffhq-config-f.pkl" # Local load, avoiding to re-download 360Mb each time
        else:
            network_pkl = self.opt.network_pkl
        print('Loading networks from "%s"...' % network_pkl)
        _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
        self.Gs = Gs
        self.Gs_syn_kwargs = dnnlib.EasyDict()
        self.Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        self.Gs_syn_kwargs.randomize_noise = False
        self.Gs_syn_kwargs.minibatch_size = 4
        self.noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
        rnd = np.random.RandomState(0)
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in self.noise_vars})



    def generate_im_from_random_seed(self, seed=22, truncation_psi=0.5):
        Gs = self.Gs
        seeds = [seed]
        noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if truncation_psi is not None:
            Gs_kwargs.truncation_psi = truncation_psi

        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            rnd = np.random.RandomState(seed)
            z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
            tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
            images = Gs.run(z, None, **Gs_kwargs)  # [minibatch, height, width, channel]
            # PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('seed%04d.png' % seed))
        return images


    def generate_im_from_z_space(self, z, truncation_psi=0.5):
        Gs = self.Gs

        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if truncation_psi is not None:
            Gs_kwargs.truncation_psi = truncation_psi  # [height, width]

        images = Gs.run(z, None, **Gs_kwargs)
        # PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('test_from_z.png'))
        return images



    def generate_im_from_w_space(self, w):

        images = self.Gs.components.synthesis.run(w, **self.Gs_syn_kwargs)
        # PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('test_from_w.png'))
        return images
















# def load_network(random_weights=False):
#     URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
#     tflib.init_tf()
#
#     with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
#         G, D, Gs = pickle.load(f)
#     if random_weights:
#         Gs.reset_vars()
#     return Gs


if __name__ == "__main__":
    Our_model = Build_model()
    # Our_model.generate_im_from_random_seed(10)
    # Our_model.generate_im_from_random_seed(50)

    rnd = np.random.RandomState(10)
    # z = rnd.randn(1, *Our_model.Gs.input_shape[1:])
    z = rnd.randn(2, 512)
    w = Our_model.Gs.components.mapping.run(z, None)
    w_avg = Our_model.Gs.get_var('dlatent_avg')

    w = w_avg + (w - w_avg) * 0.5


    Our_model.generate_im_from_w_space(w)


