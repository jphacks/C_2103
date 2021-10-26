# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response
from face import camera

app = Flask(__name__)

@app.route('/')
def index():
  pred=camera()
  return render_template('smile.html',answer=pred)

if __name__ == '__main__':
	app.run()
