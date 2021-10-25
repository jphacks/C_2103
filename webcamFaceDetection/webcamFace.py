import sys
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input

'''
kerasのバージョン　2.6.0
tensorflowのバージョン　2.6.0
'''

'''
参考
https://qiita.com/watyanabe164/items/652617c7ad577daa38d0
'''
cap = cv2.VideoCapture(0)

if cap.isOpened() is False:
    print("can not open camera")
    sys.exit()

# 評価器を読み込み
cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

# 学習済みモデルを読み込む
model = tf.keras.models.load_model('model.hdf5')

# Webカメラの映像に対して延々処理を繰り返すためwhile Trueで繰り返す。
while True:
    # VideoCaptureから1フレーム読み込む
    ret, frame = cap.read()

    # そのままの大きさだと処理速度がきついのでリサイズ
    frame = cv2.resize(frame, (int(frame.shape[1]*0.7), int(frame.shape[0]*0.7)))

    # 処理速度を高めるために画像をグレースケールに変換したものを用意
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔検出 detecctMultiScale()は検出器のルール(cascade)に従って検出した結果をfacerectに返す関数
    facerect = cascade.detectMultiScale(
        gray,
        scaleFactor=1.11,
        minNeighbors=3,
        minSize=(100, 100)
    )

    if len(facerect) != 0:
        for x, y, w, h in facerect:
            # 顔の部分
            face = frame[y: y + h, x: x + w]

            # 180x180に正規化
            reFace = cv2.resize(face, (180, 180))

    #openCV型をImage型に整形
    x = image.img_to_array(reFace)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    pred = model.predict(x)
    prediction=1-np.float(pred[0][0])

    ### デバッグ ###
    cv2.imshow('frame', reFace)# 顔領域表示
    print(prediction)
    ### ここまで ###

    if prediction < 0.4:
        print("起きろ")
    else:
        print("笑った")

    # キー入力を1ms待って、k が27（ESC）だったらBreakする
    k = cv2.waitKey(1)
    if k == 27:
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()