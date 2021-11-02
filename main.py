from flask import Flask, request, send_from_directory, send_file, abort, render_template
import flask

import os
from utils.files import dir_listing
from utils.pred import test_kepid, load_model, norm_kepid
from utils.helpers import get_robo_pred

# set the project root directory as the static folder, you can set others.
app = Flask(__name__, template_folder='./templates')

model = None
show_robo = False
prev_kepid = ''


@app.route('/lightcurves', methods=['get'])
@app.route('/lightcurves/', methods=['get'])
@app.route('/lightcurves/<path:req_path>', methods=['get'])
def list_lightcurves(req_path=''):
    # print('here', req_path)
    req_path = req_path.rstrip('/')
    BASE_DIR = 'C:/Users/User/dev/data/train/'
    return dir_listing(BASE_DIR, req_path)


@app.route('/models', methods=['get'])
@app.route('/models/', methods=['get'])
@app.route('/models/<path:req_path>', methods=['get'])
def list_models(req_path=''):
    req_path = req_path.rstrip('/')
    BASE_DIR = 'C:/Users/User/Desktop/thesis/exoplanet/saved_models'
    return dir_listing(BASE_DIR, req_path)


@app.route('/', methods=['get'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['get', 'post'])
@app.route('/predict/', methods=['get', 'post'])
def predict():
    global model
    global show_robo
    global prev_kepid
    if flask.request.method == 'GET':
        return render_template('predict.html', post=False, predictions=None, kepid=None, show_robo=show_robo,
                               prev_kepid=prev_kepid)
    elif flask.request.method == 'POST':
        kepid = flask.request.form['kepid'].strip()
        robo = flask.request.form.get('robo', type=str)
        if robo == 'on':
            show_robo = True
        else:
            show_robo = False

        if model is None:
            model = load_model('C:/Users/User/Desktop/thesis/exoplanet/saved_models/m7.h5')

        res = {}

        try:
            int(kepid)
        except:
            if prev_kepid is not None:
                kepid = prev_kepid
            else:
                return render_template('predict.html', post=False, predictions=None, kepid=None, show_robo=show_robo)

        prev_kepid = kepid
        try:
            res.update(test_kepid(model, kepid, csv_name='C:/Users/User/dev/data/q1_q17_dr24_tce_full.csv'))
        except Exception as e:
            print(e)

        # try:
        #     res.update(test_kepid(model, kepid, csv_name='C:/Users/User/dev/data/q1_q17_dr24_tce_clean.csv'))
        # except Exception as e:
        #     print(e)
        # try:
        #     res.update(test_kepid(model, kepid, csv_name='C:/Users/User/dev/data/q1_q17_dr24_tce_clean_test.csv'))
        # except Exception as e:
        #     print(e)
        print("here", res)
        robo = get_robo_pred()
        robo = robo[robo['norm_kepid'] == norm_kepid(kepid)]
        for k, v in res.items():
            pred_class = robo[robo['tce_plnt_num'] == k]['pred_class']
            if len(pred_class) == 0:
                res[k] = (f'{v*100:.3f}%', 'NaN')
            else:
                res[k] = (f'{v*100:.3f}%', 'PC' if pred_class.iloc[0] == 1 else 'Non PC')

        return render_template('predict.html', post=True, predictions=res, kepid=kepid, show_robo=show_robo,
                               prev_kepid=prev_kepid)


if __name__ == '__main__':
    app.run('0.0.0.0', 9999)
