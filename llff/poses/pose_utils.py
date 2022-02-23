import numpy as np
import os
import sys
import imageio
import skimage.transform
import shutil

from llff.poses.colmap_wrapper import run_colmap
import llff.poses.colmap_read_model as read_model


def remove_unmatched_images(realdir):
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    #This line has the list of images that are matched. Remove every other image and rerun colmap
    names = [imdata[k].name for k in imdata]
    print( 'Images #', len(names))
    
    #this list contains the list of all images in the folder
    imgs_list = os.listdir(os.path.join(realdir, 'images'))
    if len(imgs_list)==len(names):
        print("All images matched. No need to rerun colmap")
        return False
    
    unmatched_imgs = list(set(imgs_list)-set(names))
    for imgname in unmatched_imgs:
        os.remove(os.path.join(realdir, 'images/'+imgname))
    print( 'Removed unmatched images. Rerunning colmap' )
    
    return True


def load_colmap_data(realdir):
    
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    print("camera focal lengths: ", cam.params.shape)
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    #This line has the list of iamges that are matched. Remove every other image and rerun colmap
    names = [imdata[k].name for k in imdata]
    print( 'Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    print(poses.shape, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]]).shape, np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1).shape)
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm


def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind-1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape )
    
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr==1]
    print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )
    
    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        # print( i, close_depth, inf_depth )
        
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)
    print("Size of poses bounds before saving: ", save_arr.shape)
    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)
            



def minify_v0(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    def downsample(imgs, f):
        sh = list(imgs.shape)
        sh = sh[:-3] + [sh[-3]//f, f, sh[-2]//f, f, sh[-1]]
        imgs = np.reshape(imgs, sh)
        imgs = np.mean(imgs, (-2, -4))
        return imgs
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgs = np.stack([imageio.imread(img)/255. for img in imgs], 0)
    
    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
        print('Minifying', r, basedir)
        
        if isinstance(r, int):
            imgs_down = downsample(imgs, r)
        else:
            imgs_down = skimage.transform.resize(imgs, [imgs.shape[0], r[0], r[1], imgs.shape[-1]],
                                                order=1, mode='constant', cval=0, clip=True, preserve_range=False, 
                                                 anti_aliasing=True, anti_aliasing_sigma=None)
        
        os.makedirs(imgdir)
        for i in range(imgs_down.shape[0]):
            imageio.imwrite(os.path.join(imgdir, 'image{:03d}.png'.format(i)), (255*imgs_down[i]).astype(np.uint8))
            



def minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(int(100./r))
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    # imgs = [imageio.imread(f, ignoregamma=True)[...,:3]/255. for f in imgfiles]
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs


def reset_folder(basedir):
    #os.rmdir(os.path.join(basedir, 'sparse'))
    shutil.rmtree(os.path.join(basedir, 'sparse'), ignore_errors=True)
    os.remove(os.path.join(basedir, 'database.db'))
    os.remove(os.path.join(basedir, 'colmap_output.txt'))
    return

def resize_images(basedir):
    imgs_path = os.path.join(basedir, 'images')
    img_names = os.listdir(imgs_path)
    h, w = set(), set()
    for img_name in img_names:
        im = cv2.imread(os.path.join(basedir, 'images/'+img_name))
        h.add(im.shape[0])
        w.add(im.shape[1])
    if len(h)==1 and len(w)==1:
        print("All images of the same dimension")
        return
    max_h = max(h)
    max_w = max(w)
    for img_name in img_names:
        im = cv2.imread(os.path.join(basedir, 'images/'+img_name))
        diff_h = max_h - im.shape[0]
        diff_w = max_w - im.shape[1]
        im = cv2.copyMakeBorder(im.copy(),int(diff_h/2),int((diff_h+1)/2),int(diff_w/2),int((diff_w+1)/2),cv2.BORDER_CONSTANT)
        cv2.imwrite(os.path.join(basedir, 'images/'+img_name), im)
    return


def gen_poses(basedir, match_type, factors=None):
    
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print( 'Need to run COLMAP' )
        resize_images(basedir)
        run_colmap(basedir, match_type)
    else:
        print('Don\'t need to run COLMAP')
        
    print( 'Post-colmap')
    
    
    # -------------------------------------------------   Added section for unmatched images -----------------------------------------
    rerun_colmap = remove_unmatched_images(basedir)
    
    if rerun_colmap:
        reset_folder(basedir)     #Clears all files and folders except images
        files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
        if os.path.exists(os.path.join(basedir, 'sparse/0')):
            files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
        else:
            files_had = []
        if not all([f in files_had for f in files_needed]):
            print( 'Need to run COLMAP' )
            run_colmap(basedir, match_type)
        else:
            print('Don\'t need to run COLMAP')
        print( 'Post-colmap')
    # -------------------------------------------------   Added section for unmatched images -----------------------------------------
    
    
    poses, pts3d, perm = load_colmap_data(basedir)
    
    save_poses(basedir, poses, pts3d, perm)
    
    if factors is not None:
        print( 'Factors:', factors)
        minify(basedir, factors)
    
    print( 'Done with imgs2poses' )
    
    return True
    
