import azure.functions as func
from app.data import Data
from jinja2 import Environment, FileSystemLoader, select_autoescape
from app.steps import Steps

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
    sample_seed = req.params.get('sample_seed')
    if (sample_seed is None):
        sample_seed = 0

    samples = []
    for i in range(5):
        seed = 5*int(sample_seed)+i
        samples.append(data.display_sample(nb_images=1, step=Steps.TEST, seed=seed))
    
    hist1 = data.plot_values_by_step(df_name='images',
                         x='width',
                         title='Largeur des images',
                         x_label='largeur - px',
                         y_label='nombre d\'images',
                         steps=[Steps.TRAIN, Steps.VALID])
    

    template = env.get_template("dashboard.html")
    html = template.render(samples=samples, hist1=hist1)

    return func.HttpResponse(
                        html,
                        mimetype="text/html",
                 )