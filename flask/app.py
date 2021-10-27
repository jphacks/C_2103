# -*- coding: utf-8 -*-
from flask import Flask, stream_with_context, request, Response, flash
from flask import render_template, redirect
from face import get_pred
import cv2

app = Flask(__name__)

@app.route('/')
def index():
  return redirect('/stream')

def stream_template(template_name, **context):
  app.update_template_context(context)
  t = app.jinja_env.get_template(template_name)
  rv = t.stream(context)
  rv.disable_buffering()
  return rv

data=[0]
def generate():
  while True:
    data[0]=get_pred()
    yield str(data[0])

@app.route('/stream')
def stream_feed():
  rows=generate()
  return Response(stream_with_context(stream_template('smile.html',rows=rows)))

if __name__ == '__main__':
  app.run()
