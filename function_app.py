import azure.functions as func
from app.data import Data
from jinja2 import Environment, FileSystemLoader, select_autoescape
from app.steps import Steps

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape()
)

@app.route(route="/update_data", methods=["GET"])
def update_data(req: func.HttpRequest) -> func.HttpResponse:
    """Route pour preparer les data
    """
    Data(force_extract=True, mode='html')
    return func.HttpResponse(
                        "Ok",
                        mimetype="text/html",
                 )

@app.route(route="/", methods=["GET"])
def home(req: func.HttpRequest) -> func.HttpResponse:
    """Route pour afficher le dashboard
    """
    data = Data(mode='html')
    hist1 = data.plot_values_by_step(df_name='images',
                         x='width',
                         title='Largeur des images',
                         x_label='largeur - px',
                         y_label='nombre d\'images',
                         steps=[Steps.TRAIN, Steps.VALID])
    

    template = env.get_template("home.html")
    html = template.render(users=['titi', 'tata'], hist1=hist1)

    return func.HttpResponse(
                        html,
                        mimetype="text/html",
                 )