# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response
from face import get_pred

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('smile.html')

def prediction():
    while True:
        pred = get_pred()
        yield pred

if __name__ == '__main__':
  app.run()
