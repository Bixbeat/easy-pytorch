from PIL import Image
from os.path import basename

def downsample_image(in_img_path, out_img_path, resolution):
    img = Image.open(in_img_path)
    out_img = img.resize(img,resolution, interpolation=Image.LANCZOS)
    out_img.save(out_img_path)

if __name__ == '__main__':
    import glob

    img_path = "LR/"
    glob_expr = img_path + "*.tif"

    imgs = glob.glob(glob_expr)

    for image in imgs:
        img_name = basename(image)
        downsample_image(image, 'lowres/'+img_name)
