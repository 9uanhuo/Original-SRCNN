import cv2
import random
import glob


# 任意のフレーム数を切り出すプログラム
class DataCreate:
    def __init__(self):
        self.num = 0
        self.mag = 2

    # 任意のフレーム数を切り出すプログラム
    def data_create(self,
                    img_path,  # 切り取る動画が入ったファイルのpath
                    data_number,  # データセットの生成数
                    cut_frame,  # 1枚の画像から生成するデータセットの数
                    hr_height,  # HRの保存サイズ
                    hr_width):

        lr_height = hr_height  # 低解像度画像のsize = 高解像度のsize
        lr_width = hr_width

        low_data_list = []  # 生成したデータを格納するListの作成
        high_data_list = []
        low_data_list_cb = []  # 生成したデータを格納するListの作成
        high_data_list_cb = []
        low_data_list_cr = []  # 生成したデータを格納するListの作成
        high_data_list_cr = []
        path = img_path + "/*"
        files = glob.glob(path)

        while self.num < data_number:
            photo_num = random.randint(0, len(files) - 1)
            img = cv2.imread(files[photo_num])
            height, width = img.shape[:2]

            if hr_height > height or hr_width > width:
                break

            color_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            gray = color_img[:, :, 0]
            cr = color_img[:, :, 1]
            cb = color_img[:, :, 2]
            bicubic_img = cv2.resize(gray, (int(width // self.mag), int(height // self.mag)),
                                     interpolation=cv2.INTER_CUBIC)
            bicubic_img = cv2.resize(bicubic_img, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)

            bicubic_img_cr = cv2.resize(cr, (int(width // self.mag), int(height // self.mag)),
                                        interpolation=cv2.INTER_CUBIC)
            bicubic_img_cr = cv2.resize(bicubic_img_cr, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)

            bicubic_img_cb = cv2.resize(cb, (int(width // self.mag), int(height // self.mag)),
                                        interpolation=cv2.INTER_CUBIC)
            bicubic_img_cb = cv2.resize(bicubic_img_cb, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)

            for i in range(cut_frame):
                ram_h = random.randint(0, height - lr_height)
                ram_w = random.randint(0, width - lr_width)

                lr_img = bicubic_img[ram_h: ram_h + lr_height, ram_w: ram_w + lr_width]
                high_img = gray[ram_h: ram_h + hr_height, ram_w: ram_w + hr_width]

                lr_img_cr = bicubic_img_cr[ram_h: ram_h + lr_height, ram_w: ram_w + lr_width]
                high_img_cr = cr[ram_h: ram_h + hr_height, ram_w: ram_w + hr_width]

                lr_img_cb = bicubic_img_cb[ram_h: ram_h + lr_height, ram_w: ram_w + lr_width]
                high_img_cb = cb[ram_h: ram_h + hr_height, ram_w: ram_w + hr_width]

                low_data_list.append(lr_img)
                high_data_list.append(high_img)
                low_data_list_cr.append(lr_img_cr)
                high_data_list_cr.append(high_img_cr)
                low_data_list_cb.append(lr_img_cb)
                high_data_list_cb.append(high_img_cb)

                self.num += 1

                if self.num == data_number:
                    break

        return low_data_list, high_data_list, low_data_list_cr, high_data_list_cr, low_data_list_cb, high_data_list_cb
