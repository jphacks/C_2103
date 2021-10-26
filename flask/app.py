# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response
from face import get_pred

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('smile.html')

def prediction():
    while True:
        pred = str(get_pred())
        yield (b'--frame\r\n'
                  b'Content-type: text/html\n' + pred + b'\r\n')

@app.route('/video_feed')
def video_feed():
   #imgタグに埋め込まれるResponseオブジェクトを返す
   return Response(prediction(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
  app.run()
