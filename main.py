import model
import data_create
import argparse
import os
import cv2
import numpy as np
import tensorflow as tf


def check_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
    else:
        print("Not enough GPU hardware devices available")


def output_img(i, i_cr, i_cb):
    img = tf.keras.preprocessing.image.img_to_array(
        tf.reshape(i[p] * 255, [args.test_height, args.test_width]))
    img_cr = tf.keras.preprocessing.image.img_to_array(
        tf.reshape(i_cr[p] * 255, [args.test_height, args.test_width]))
    img_cb = tf.keras.preprocessing.image.img_to_array(
        tf.reshape(i_cb[p] * 255, [args.test_height, args.test_width]))
    img = img.astype(np.uint8)
    img_cr = img_cr.astype(np.uint8)
    img_cb = img_cb.astype(np.uint8)
    output = cv2.merge([img, img_cr, img_cb])
    dest = cv2.cvtColor(output, cv2.COLOR_YCrCb2BGR)
    return dest


def ssim(img1, img2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) *
                                                            (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssim_list = []
            for i in range(3):
                ssim_list.append(ssim(img1, img2))
            return np.array(ssim_list).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow SRCNN Example')

    parser.add_argument('--train_height', type=int, default=33, help="Train data size(height)")
    parser.add_argument('--train_width', type=int, default=33, help="Train data size(width)")
    parser.add_argument('--test_height', type=int, default=700, help="Test data size(height)")
    parser.add_argument('--test_width', type=int, default=700, help="Test data size(width)")
    parser.add_argument('--train_dataset_num', type=int, default=10000, help="Number of train datasets to generate")
    parser.add_argument('--test_dataset_num', type=int, default=5, help="Number of test datasets to generate")
    parser.add_argument('--train_cut_num', type=int, default=10,
                        help="Number of train data to be generated from a single image")
    parser.add_argument('--test_cut_num', type=int, default=1,
                        help="Number of test data to be generated from a single image")
    parser.add_argument('--train_path', type=str, default="./dataset/DIV2K_train_HR",
                        help="The path containing the train image")
    parser.add_argument('--test_path', type=str, default="./dataset/DIV2K_valid_HR",
                        help="The path containing the test image")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning_rate")
    parser.add_argument('--BATCH_SIZE', type=int, default=32, help="Training batch size")
    parser.add_argument('--EPOCHS', type=int, default=300, help="Number of epochs to train for")


    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, 1, name=None)


    parser.add_argument('--mode', type=str, default='train_model',
                        help='train_data_create, test_data_create, train_model, evaluate')

    args = parser.parse_args()

    if args.mode == 'train_data_create':  # 学習用データセットの生成
        create_data = data_create.DataCreate()
        train_x, train_y, t_x_cr, t_y_cr, t_x_cb, t_y_cb = create_data.data_create(args.train_path,
                                                                                   # 切り取る動画のpath
                                                                                   args.train_dataset_num,
                                                                                   # データセットの生成数
                                                                                   args.train_cut_num,
                                                                                   # 1枚の画像から生成するデータの数
                                                                                   args.train_height,
                                                                                   # 保存サイズ
                                                                                   args.train_width)
        path = "train_data_list"
        path_cr = "train_data_list_cr"
        path_cb = "train_data_list_cb"
        np.savez(path, train_x, train_y)
        np.savez(path_cr, t_x_cr, t_y_cr)
        np.savez(path_cb, t_x_cb, t_y_cb)

    elif args.mode == 'test_data_create':  # 評価用データセットの生成
        create_data = data_create.DataCreate()
        test_x, test_y, e_x_cr, e_y_cr, e_x_cb, e_y_cb = create_data.data_create(args.test_path,
                                                                                 args.test_dataset_num,
                                                                                 args.test_cut_num,
                                                                                 args.test_height,
                                                                                 args.test_width)

        path = "test_data_list"
        path_cr = "test_data_list_cr"
        path_cb = "test_data_list_cb"
        np.savez(path, test_x, test_y)
        np.savez(path_cr, e_x_cr, e_y_cr)
        np.savez(path_cb, e_x_cb, e_y_cb)

    elif args.mode == "train_model":  # 学習
        check_gpu()
        npz = np.load("train_data_list.npz")
        train_x = npz["arr_0"]
        train_y = npz["arr_1"]

        train_x = tf.convert_to_tensor(train_x, np.float32)
        train_y = tf.convert_to_tensor(train_y, np.float32)

        train_x /= 255
        train_y /= 255

        train_model = model.srcnn()

        optimizers = tf.keras.optimizers.Adam(lr=args.learning_rate)
        train_model.compile(loss="mean_squared_error",
                            optimizer=optimizers,
                            metrics=[psnr])

        train_model.fit(train_x,
                        train_y,
                        epochs=args.EPOCHS,
                        verbose=2,
                        batch_size=args.BATCH_SIZE)

        train_model.save("SRCNN_model.h5")

    elif args.mode == "evaluate":  # 評価
        check_gpu()

        result_path = "result"
        os.makedirs(result_path, exist_ok=True)

        npz = np.load("test_data_list.npz", allow_pickle=True)
        npz_cr = np.load("test_data_list_cr.npz", allow_pickle=True)
        npz_cb = np.load("test_data_list_cb.npz", allow_pickle=True)

        test_x = npz["arr_0"]
        test_y = npz["arr_1"]
        test_x_cr = npz_cr["arr_0"]
        test_y_cr = npz_cr["arr_1"]
        test_x_cb = npz_cb["arr_0"]
        test_y_cb = npz_cb["arr_1"]

        test_x = tf.convert_to_tensor(test_x, np.float32)
        test_y = tf.convert_to_tensor(test_y, np.float32)
        test_x_cr = tf.convert_to_tensor(test_x_cr, np.float32)
        test_y_cr = tf.convert_to_tensor(test_y_cr, np.float32)
        test_x_cb = tf.convert_to_tensor(test_x_cb, np.float32)
        test_y_cb = tf.convert_to_tensor(test_y_cb, np.float32)

        test_x /= 255
        test_y /= 255
        test_x_cr /= 255
        test_y_cr /= 255
        test_x_cb /= 255
        test_y_cb /= 255

        path = "SRCNN_model.h5"

        if os.path.exists(path):
            model = tf.keras.models.load_model(path, custom_objects={'psnr': psnr})
            pred = model.predict(test_x, batch_size=1)
            pred_output = []
            for p in range(len(test_y)):
                pred[p][pred[p] > 1] = 1
                pred[p][pred[p] < 0] = 0
                ps_pred = psnr(tf.reshape(test_y[p], [args.test_height, args.test_width, 1]), pred[p])
                ps_bicubic = psnr(tf.reshape(test_y[p], [args.test_height, args.test_width, 1]),
                                  tf.reshape(test_x[p], [args.test_height, args.test_width, 1]))

                low_dest = output_img(test_x, test_x_cr, test_x_cb)
                cv2.imwrite(result_path + "/" + str(p) + "_low" + ".jpg", low_dest)  # LR

                high_dest = output_img(test_y, test_y_cr, test_y_cb)
                cv2.imwrite(result_path + "/" + str(p) + "_high" + ".jpg", high_dest)  # HR

                pred_dest = output_img(pred, test_x_cr, test_x_cb)
                cv2.imwrite(result_path + "/" + str(p) + "_pred" + ".jpg", pred_dest)  # pred

                ss_pred = calculate_ssim(pred_dest, high_dest)
                ss_bicubic = calculate_ssim(low_dest, high_dest)

                print("num:{}".format(p))
                print("psnr_pred:{}".format(ps_pred))
                print("psnr_bicubic:{}".format(ps_bicubic))
                print("ssim_pred:{}".format(ss_pred))
                print("ssim_bicubic:{}".format(ss_bicubic))
