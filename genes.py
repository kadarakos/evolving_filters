from skimage import filters
import numpy as np
import skimage
from skimage.exposure import equalize_hist
from PIL import Image, ImageFilter


def identity(image):
    '''Do nothing.'''
    return image


def individual_to_string(ind):
    '''Print the names of the list of functions making up the individual.'''
    for i in ind:
        try:
            print i.func_name
        except:
            print (i.func.func_name, i.keywords)

def split_image_into_channels(image):
    '''Return red, green and blue channels given image.'''
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    return red_channel, green_channel, blue_channel

def merge_channels(red_channel, green_channel, blue_channel):
    '''Stack red, green and blue channels together.'''
    return np.stack([red_channel, green_channel, blue_channel], axis=2)

def PIL_to_array(im):
    '''Convert PIL image to float numpy array.'''
    (width, height) = im.size
    img = list(im.getdata())
    img = np.array(img, dtype='uint8')
    img = img.reshape((height, width, 3))
    return skimage.img_as_float(img)

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


def filter_min(image, size):
    '''Min filter using PIL.'''
    pil_img = Image.fromarray(skimage.img_as_ubyte(image))
    fil = ImageFilter.MinFilter(size)
    im = pil_img.filter(fil)
    return PIL_to_array(im)


def filter_max(image, size):
    '''Max filter using PIL.'''
    pil_img = Image.fromarray(skimage.img_as_ubyte(image))
    fil = ImageFilter.MaxFilter(size)
    im = pil_img.filter(fil)
    return PIL_to_array(im)


def filter_mode(image, size):
    '''Mode filter using PIL.'''
    pil_img = Image.fromarray(skimage.img_as_ubyte(image))
    fil = ImageFilter.ModeFilter(size)
    im = pil_img.filter(fil)
    return PIL_to_array(im)


def filter_median(image, size):
    '''Median filter using PIL.'''
    pil_img = Image.fromarray(skimage.img_as_ubyte(image))
    fil = ImageFilter.MedianFilter(size)
    im = pil_img.filter(fil)
    return PIL_to_array(im)



gene_pool = [sharpen, adjust_r, adjust_b, adjust_r,
        identity, equalize_r, equalize_g, equalize_b,
        binarize_r, binarize_g, binarize_b, filter_max,
        filter_min, filter_mode, filter_median]
