import azure.functions as func
from app.data import Data
from app.models import Models
from jinja2 import Environment, FileSystemLoader, select_autoescape
from app.steps import Steps
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape()
)

data = Data(mode='html')
models = Models(data=data, prepare_for_train=False)
roll = 900

@app.route(route="/explore", methods=["GET"])
def explore(req: func.HttpRequest) -> func.HttpResponse:
    """Route pour afficher l'exploration
    """
    indexes = data.get_indexes(step=Steps.TEST, start=0, quantity=3, roll=roll)
    template = env.get_template("exploration/explore.html")
    html = template.render(indexes=indexes)

    return func.HttpResponse(
                        html,
                        mimetype="text/html",
                 )

@app.route(route="/perfs", methods=["GET"])
def performances(req: func.HttpRequest) -> func.HttpResponse:
    """Route pour afficher les performances modèles
    """
    template = env.get_template("performances/perfs.html")
    html = template.render()

    return func.HttpResponse(
                        html,
                        mimetype="text/html",
                 )

@app.route(route="/predict", methods=["GET"])
def predict(req: func.HttpRequest) -> func.HttpResponse:
    """Route pour afficher une prédiction
    """
    indexes = data.get_indexes(step=Steps.TEST, start=0, quantity=10, roll=roll)
    template = env.get_template("prediction/predict.html")
    html = template.render(indexes=indexes)
    return func.HttpResponse(
                        html,
                        mimetype="text/html",
                 )

@app.route(route="/predict_image", methods=["GET"])
def predict_image(req: func.HttpRequest) -> func.HttpResponse:
    index = req.params.get('index')
    model = req.params.get('model')
    return func.HttpResponse(
                        models.predict(index=index, model=model),
                        mimetype="image/png",
                 )

@app.route(route="/images-indexes", methods=["GET"])
def images_indexes(req: func.HttpRequest) -> func.HttpResponse:
    start_image = req.params.get('start_image')
    quantity = req.params.get('quantity')
    indexes = data.get_indexes(step=Steps.TEST, start=int(start_image), quantity=int(quantity), roll=roll)
    return func.HttpResponse(
                        json.dumps({'indexes': indexes}),
                        mimetype="application/json",
                 )

@app.route(route="/image", methods=["GET"])
def image(req: func.HttpRequest) -> func.HttpResponse:
    index = req.params.get('index')
    return func.HttpResponse(
                        data.display_sample(indexes=[index], step=Steps.TEST),
                        mimetype="image/png",
                 )

@app.route(route="/image-table", methods=["GET"])
def image_table(req: func.HttpRequest) -> func.HttpResponse:
    index = req.params.get('index')
    [_,_,_,table] = data.get(indexes=[index], step=Steps.TEST),
    return func.HttpResponse(
                        json.dumps({'table': table}),
                        mimetype="application/json"
                 )

@app.route(route="/values-by-step", methods=["GET"])
def values_by_step(req: func.HttpRequest) -> func.HttpResponse:
    x = req.params.get('x')
    texts = values_by_step_texts(req)
    img = data.plot_values_by_step(df_name='images',
                                   x=x,
                                   title=texts['title'],
                                   x_label=texts['x_label'],
                                   y_label='nombre d\'images',
                                   steps=steps(req))
    return func.HttpResponse(
                        img,
                        mimetype="image/png",
                 )

@app.route(route="/values-by-step-title", methods=["GET"])
def values_by_step_title(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(
                        values_by_step_texts(req)['title'],
                        mimetype="text/html",
                 )

@app.route(route="/hist-by-step", methods=["GET"])
def hist_by_step(req: func.HttpRequest) -> func.HttpResponse:
    x = req.params.get('x')
    bins = req.params.get('bins')
    texts = hist_by_step_texts(req)
    img = data.plot_hist_by_step(df_name='boxes',
                                 x=x,
                                 bins=int(bins),
                                 title=texts['title'],
                                 x_label=texts['x_label'],
                                 y_label='nombre d\'images',
                                 steps=steps(req))
    return func.HttpResponse(
                        img,
                        mimetype="image/png",
                 )

@app.route(route="/hist-by-step-title", methods=["GET"])
def hist_by_step_title(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(
                        hist_by_step_texts(req)['title'],
                        mimetype="text/html",
                 )

@app.route(route="/valid-boxes-by-step", methods=["GET"])
def valid_boxes_by_step(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(
                        data.plot_pie_valid_boxes_by_step(steps=steps(req)),
                        mimetype="image/png",
                 )

@app.route(route="/boxes-per-image-by-step", methods=["GET"])
def boxes_per_image_by_step(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(
                        data.plot_count_boxes_per_images_by_step(steps=steps(req)),
                        mimetype="image/png",
                 )

@app.route(route="/boxes-in-same-cell-by-step", methods=["GET"])
def boxes_in_same_cell_by_step(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(
                        data.plot_box_in_same_cell(steps=steps(req)),
                        mimetype="image/png",
                 )

def steps(req: func.HttpRequest):
    steps = []
    if (req.params.get('train') == 'true'):
        steps.append(Steps.TRAIN)
    if (req.params.get('valid') == 'true'):
        steps.append(Steps.VALID)
    if (req.params.get('test') == 'true'):
        steps.append(Steps.TEST)
    if len(steps) == 0:
        steps = Steps.ALL
    return steps

def values_by_step_texts(req: func.HttpRequest):
    x = req.params.get('x')
    
    title = x
    x_label = x
    if x == 'width':
        title = 'Largeur des images'
        x_label = 'largeur - px'
    elif x == 'height':
        title = 'Hauteur des images'
        x_label = 'hauteur - px'
    elif x == 'ratio':
        title = 'Ratio des images'
        x_label = 'ratio'

    return {'title': title, 'x_label': x_label}

def hist_by_step_texts(req: func.HttpRequest):
    x = req.params.get('x')
    
    title = x
    x_label = x
    if x == 'x':
        title = 'Répartition de la position en X des boites'
        x_label = 'répartition normalisée en X'
    elif x == 'y':
        title = 'Répartition de la position en X des boites'
        x_label = 'répartition normalisée en X'
    elif x == 'width':
        title = 'Répartition de la largeur des boites'
        x_label = 'répartition normalisée de la largeur'
    elif x == 'height':
        title = 'Répartition de la hauteur des boites'
        x_label = 'répartition normalisée de la hauteur'
    elif x == 'surface':
        title = 'Répartition de la surface des boites'
        x_label = 'répartition normalisée de la surface'

    return {'title': title, 'x_label': x_label}
