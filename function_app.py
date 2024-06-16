import azure.functions as func
import logging
from app.data import Data
import json
from jinja2 import Environment, FileSystemLoader, select_autoescape

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape()
)

@app.route(route="/prepare_data", methods=["GET"])
def home(req: func.HttpRequest) -> func.HttpResponse:
    """Route pour preparer les data
    """
    data = Data(force_extract=True)
    data.prepare_data()
    return func.HttpResponse(
                        "Ok",
                        mimetype="text/html",
                 )

@app.route(route="/", methods=["GET"])
def home(req: func.HttpRequest) -> func.HttpResponse:
    """Route pour afficher le dashboard
    """
    data = Data(mode='html')
    template = env.get_template("home.html")
    html = template.render(users=['titi', 'tata'])

    # return f"<img src='data:image/png;base64,{data}'/>"
    png_hist_by_step = data.plot_hist_by_step()

    return func.HttpResponse(
                        html,
                        mimetype="text/html",
                 )