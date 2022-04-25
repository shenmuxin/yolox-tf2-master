import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from config import cfg
import sys
sys.path.append(cfg.FILE.ABSPATH)          #   修改为本地的文件路径


from yolox_Detection import YOLOXDetect

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    

class Prediction:
    _defaults = {
        #   mode可选video, image, image_dir, fps, test
        'mode'              :   'video',
        #   video_path为空表示调用摄像头进行检测
        'video_path'        :   '',
        #   video检测保存路径
        'video_save_path'   :   '',
        #   video检测保存的fps
        'video_fps'         :   25.0,
        'test_interval'     :   1000,
        #   指定多图片检测的路径
        'dir_origin_path'   :   '',
        #   指定多图检测的保存路径
        'dir_save_path'     :   '',
        #   是否对检测目标进行裁剪
        'crop'              :   False
    }


    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value) 
        
        self.yolo = YOLOXDetect()



    def start_detection(self):

        if self.mode == "image":  # 单张图片检测
            import os

            while True:
                img = input('Input image filename:')
                if img == 'stop': break
                try:
                    image = Image.open(img)
                except:
                    print('Open Error! Try again!')
                    continue
                else:
                    r_image = self.yolo.detect_image(image, self.crop)
                    r_image.show()

                    filename = input('Input save name')
                    if filename != 'q':
                        r_image.save(os.path.join(r'D:\fig', filename + '.jpg'))

        elif self.mode == "video":  # 视频检测
            capture = cv2.VideoCapture(self.video_path if self.video_path else 0)
            if self.video_save_path!="":
                fourcc  = cv2.VideoWriter_fourcc(*'XVID')
                size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                out     = cv2.VideoWriter(self.video_save_path, fourcc, self.video_fps, size)
            ret, frame = capture.read()
            if not ret:
                raise ValueError("未能正确读取摄像头")
            fps = 0.0
            while(True):
                t1 = time.time()
                #   读取某一帧
                ret, frame = capture.read()
                #   读取完成退出
                if not ret:
                    break
                #   BGRtoRGB,满足PIL显示格式
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                #   转变成Image
                frame = Image.fromarray(frame.astype('uint8'))
                #   进行检测,并转换为ndarry
                frame = np.array(self.yolo.detect_image(frame, self.crop))
                #   RGBtoBGR满足opencv显示格式
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                #   与之前的fps做累计取平均值
                fps  = ( fps + (2./(time.time()-t1)) ) / 2

                print("fps = %.2f"%(fps))
                
                text_org = (0, int(5e-2*capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                frame = cv2.putText(frame, "fps:%.2f"%(fps), text_org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("video", frame)
                if self.video_save_path!="":
                    out.write(frame)
                #   ESC退出
                if cv2.waitKey(1) == 27:  
                    capture.release()
                    break
            print("Video Detection Done!")
            capture.release()
            if self.video_save_path!="":
                print("Save processed video to the path :" + self.video_save_path)
                out.release()
            cv2.destroyAllWindows()

        elif self.mode == "fps":  # fps测试
            img = Image.open('./dataset/img/street.jpg')
            tact_time = self.yolo.get_FPS(img, self.test_interval)
            print('-'*50)
            print('Detect a image per {0:.3f} second\t|\tFPS: {1:.3f}'.format(tact_time, 1/tact_time))
            print('-'*50)

        elif self.mode == "image_dir":  # 多图检测
            import os
            from tqdm import tqdm
            img_names = os.listdir(self.dir_origin_path)
            for img_name in tqdm(img_names, desc='Processing'):
                if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    image_path  = os.path.join(self.dir_origin_path, img_name)
                    image       = Image.open(image_path)
                    r_image     = self.yolo.detect_image(image, self.crop)
                    #   多图保存
                    if not os.path.exists(self.dir_save_path):
                        os.makedirs(self.dir_save_path)
                    r_image.save(os.path.join(self.dir_save_path, img_name))
        



if __name__ == "__main__":
    prediction = Prediction()
    prediction.start_detection()

