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
    force_extract = req.params.get('force_extract') == 'true'
    force_dfs = req.params.get('force_dfs') == 'true'

    Data(force_dfs=force_dfs, force_extract=force_extract, mode='html')
    return func.HttpResponse(
                        "Ok",
                        mimetype="text/html",
                 )

@app.route(route="/dashboard", methods=["GET"])
def dashboard(req: func.HttpRequest) -> func.HttpResponse:
    """Route pour afficher le dashboard
    """
    data = Data(mode='html')
    hist1 = data.plot_values_by_step(df_name='images',
                         x='width',
                         title='Largeur des images',
                         x_label='largeur - px',
                         y_label='nombre d\'images',
                         steps=[Steps.TRAIN, Steps.VALID])
    

    template = env.get_template("dashboard.html")
    html = template.render(hist1=hist1)

    return func.HttpResponse(
                        html,
                        mimetype="text/html",
                 )