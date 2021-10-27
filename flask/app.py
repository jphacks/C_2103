# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response, redirect,stream_with_context
from face import get_pred
from time import sleep

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
data = [0] 
def generate():
    while True:
      pred = get_pred()
      data[0]=pred
      yield  str(data[0])

@app.route('/stream')
def stream_view():

  rows = generate()
  return Response(stream_with_context(stream_template('smile.html', rows=rows)))


if __name__ == '__main__':
  app.run()