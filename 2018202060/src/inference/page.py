from flask import Flask, render_template, request, redirect
import os
import inference

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/'


@app.route('/', methods = ['GET','POST'])
def input():
   if request.method == 'GET':
      return render_template('input.html')
   else:
      img_file = request.files['img_file']
      data_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
      img_file.save(data_path)
      
      result = inference.classify(data_path)
      #result = inference.test()

      return render_template('result.html', result = result, data_path = data_path)



if __name__ == '__main__':
   app.run()


