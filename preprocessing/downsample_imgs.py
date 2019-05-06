import cv2
from os.path import basename

def downsample_image(in_img_path, out_img_path, scale_factor=0.5):
    img = cv2.imread(in_img_path)
    out_img = cv2.resize(img,(38, 38), interpolation = cv2.INTER_AREA)
    cv2.imwrite(out_img_path, out_img)

if __name__ == '__main__':
    import glob

    img_path = "LR/"
    glob_expr = img_path + "*.tif"

    imgs = glob.glob(glob_expr)

    for image in imgs:
        img_name = basename(image)
        downsample_image(image, 'lowres/'+img_name)
