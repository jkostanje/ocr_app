from flask_restx import Namespace

ns = Namespace('Predict', path='/')


from sample_project.predict.v1 import views  # noqa
