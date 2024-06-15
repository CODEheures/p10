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

@app.route(route="/", methods=["GET"])
def home(req: func.HttpRequest) -> func.HttpResponse:
    """Route pour afficher le dashboard
    """
    template = env.get_template("home.html")
    html = template.render(users=['titi', 'tata'])
    return func.HttpResponse(
                        html,
                        mimetype="text/html",
                 )