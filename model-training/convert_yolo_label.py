import os
from PIL import Image

# modify the directories and class file to fit
datapath = './UECFOOD256'

def convert_yolo_bbox(img_size, box):
    # img_bbox file is [0:img] [1:left X] [2:bottom Y] [3:right X] [4:top Y]
    dw = 1./img_size[0]
    dh = 1./img_size[1]
    x = (int(box[1]) + int(box[3]))/2.0
    y = (int(box[2]) + int(box[4]))/2.0
    w = abs(int(box[3]) - int(box[1]))
    h = abs(int(box[4]) - int(box[2]))
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    # Yolo bbox is center x, y and width, height
    return (x,y,w,h)

def generate_bbox_file(datapath):
    for dir in os.listdir(datapath):
        if len(dir.split('.')) == 1:
            pics_dir = os.path.join(datapath, dir)
            bb_filename = os.path.join(pics_dir, 'bb_info.txt')
            with open(bb_filename) as fp:
                for line in fp.readlines():
                    # img_bbox file is [0:img] [1:left X] [2:bottom Y] [3:right X] [4:top Y]
                    img_bbox = line.strip('\n').split(' ')
                    if img_bbox[0] != 'img':
                        image_filename = os.path.join(datapath, dir, img_bbox[0]+'.jpg')
                        yolo_label_filename = os.path.join('./labels', img_bbox[0]+'.txt')
                        with open(yolo_label_filename, 'w') as f:
                            img = Image.open(image_filename)
                            yolo_bbox = convert_yolo_bbox(img.size, img_bbox)
                            if (yolo_bbox[2] > 1) or (yolo_bbox[3] > 1):
                                print("image %s bbox is " %(image_filename) + ' '.join(map(str, yolo_bbox)))
                            f.write(dir + ' ' + ' '.join(map(str, yolo_bbox)) + '\n')
                            img.save(os.path.join('./images', img_bbox[0]+'.jpg'))
                            img.close()
                            f.close()
                fp.close()

generate_bbox_file(datapath)
