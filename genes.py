from skimage import filters
import numpy as np
import skimage
from skimage.exposure import equalize_hist


def identity(image):
    '''Do nothing.'''
    return image

def split_image_into_channels(image):
    '''Return red, green and blue channels given image.'''
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    return red_channel, green_channel, blue_channel

def merge_channels(red_channel, green_channel, blue_channel):
    '''Stack red, green and blue channels together.'''
    return np.stack([red_channel, green_channel, blue_channel], axis=2)

def sharpen(image, a, b, sigma):
    '''Apply gaussian sharpening or blurring to image.'''
    blurred = skimage.filters.gaussian(image, sigma=sigma, multichannel=True)
    sharper = np.clip(image * a - blurred * b, 0, 1.0)
    return sharper

def channel_adjust(channel, values):
    '''Screw around with interpolating color channel.'''
    # flatten
    orig_size = channel.shape
    flat_channel = channel.flatten()
    adjusted = np.interp(
        flat_channel,
        np.linspace(0, 1, len(values)),
        values)
    # put back into image form
    return adjusted.reshape(orig_size)

def adjust_r(image, values):
    '''Adjust red channel.'''
    r, g, b = split_image_into_channels(image)
    r_adjusted = channel_adjust(r, values)
    merged = merge_channels(r_adjusted, g, b)
    return merged

def adjust_g(image, values):
    '''Adjust green channel.'''
    r, g, b = split_image_into_channels(image)
    g_adjusted = channel_adjust(g, values)
    merged = merge_channels(r, g_adjusted, b)
    return merged

def adjust_b(image, values):
    '''Adjust blue channel.'''
    r, g, b = split_image_into_channels(image)
    b_adjusted = channel_adjust(b, values)
    merged = merge_channels(r, g, b_adjusted)
    return merged

def equalize_r(image):
    '''Equalize red channel.'''
    r, g, b = split_image_into_channels(image)
    return merge_channels(equalize_hist(r), g, b)

def equalize_g(image):
    '''Equalize green channel.'''
    r, g, b = split_image_into_channels(image)
    return merge_channels(r, equalize_hist(g), b)

def equalize_b(image):
    '''Equalize blue channel.'''
    r, g, b = split_image_into_channels(image)
    return merge_channels(r, g, equalize_hist(b))

def binarize_r(image):
    '''Binarize red channel.'''
    r, g, b = split_image_into_channels(image)
    b_r = np.where(r > np.mean(r),1.0,0.0)
    return merge_channels(b_r, g, b)

def binarize_g(image):
    '''Binarize green channel.'''
    r, g, b = split_image_into_channels(image)
    b_g = np.where(g > np.mean(g),1.0,0.0)
    return merge_channels(r, g, b)

def binarize_b(image):
    '''Binarize blue channel.'''
    r, g, b = split_image_into_channels(image)
    b_b = np.where(b > np.mean(b),1.0,0.0)
    return merge_channels(r, g, b_b)

gene_pool = [sharpen, adjust_r, adjust_b, adjust_r,
        identity, equalize_r, equalize_g, equalize_b,
        binarize_r, binarize_g, binarize_b]
