import azure.functions as func
from app.data import Data
from jinja2 import Environment, FileSystemLoader, select_autoescape
from app.steps import Steps
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape()
)

data = Data(mode='html')

@app.route(route="/dashboard", methods=["GET"])
def dashboard(req: func.HttpRequest) -> func.HttpResponse:
    """Route pour afficher le dashboard
    """
    indexes = data.get_indexes(step=Steps.TEST, start=0, quantity=3, roll=135)
    template = env.get_template("dashboard.html")
    html = template.render(indexes=indexes)

    return func.HttpResponse(
                        html,
                        mimetype="text/html",
                 )

@app.route(route="/images-indexes", methods=["GET"])
def images_indexes(req: func.HttpRequest) -> func.HttpResponse:
    start_image = req.params.get('start_image')
    indexes = data.get_indexes(step=Steps.TEST, start=int(start_image), quantity=3, roll=135)
    return func.HttpResponse(
                        json.dumps({'indexes': indexes}),
                        mimetype="application/json",
                 )

@app.route(route="/sample_image", methods=["GET"])
def sample_image(req: func.HttpRequest) -> func.HttpResponse:
    index = req.params.get('index')
    return func.HttpResponse(
                        data.display_sample(indexes=[index], step=Steps.TEST),
                        mimetype="image/png",
                 )

@app.route(route="/values-by-step", methods=["GET"])
def values_by_step(req: func.HttpRequest) -> func.HttpResponse:
    x = req.params.get('x')
    steps = []
    if (req.params.get('train') == 'true'):
        steps.append(Steps.TRAIN)
    if (req.params.get('valid') == 'true'):
        steps.append(Steps.VALID)
    if (req.params.get('test') == 'true'):
        steps.append(Steps.TEST)
    if len(steps) == 0:
        steps = Steps.ALL
    
    title = x
    x_label = x
    if x == 'width':
        title = 'Largeur des images'
        x_label = 'largeur - px'

    img = data.plot_values_by_step(df_name='images',
                         x=x,
                         title=title,
                         x_label=x_label,
                         y_label='nombre d\'images',
                         steps=steps)
    return func.HttpResponse(
                        img,
                        mimetype="image/png",
                 )