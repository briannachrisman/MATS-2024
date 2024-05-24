try:
    import google.colab # type: ignore
    from google.colab import output
    COLAB = True
    #%pip install sae-lens==1.3.0 transformer-lens==1.17.0
    #%pip install --upgrade sae-lens
except:
    COLAB = False
    from IPython import get_ipython # type: ignore
    ipython = get_ipython(); assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# Standard imports
import os
import torch
from tqdm import tqdm
import plotly.express as px

from sae_vis.data_config_classes import SaeVisConfig
from sae_vis.data_storing_fns import SaeVisData

# Imports for displaying vis in Colab / notebook
import webbrowser
import http.server
import socketserver
import threading
PORT = 8000

torch.set_grad_enabled(False);




def display_vis_inline(filename: str, height: int = 850):
    '''
    Displays the HTML files in Colab. Uses global `PORT` variable defined in prev cell, so that each
    vis has a unique port without having to define a port within the function.
    '''
    if not(COLAB):
        webbrowser.open(filename);

    else:
        global PORT

        def serve(directory):
            os.chdir(directory)

            # Create a handler for serving files
            handler = http.server.SimpleHTTPRequestHandler

            # Create a socket server with the handler
            with socketserver.TCPServer(("", PORT), handler) as httpd:
                print(f"Serving files from {directory} on port {PORT}")
                httpd.serve_forever()

        thread = threading.Thread(target=serve, args=("/content",))
        thread.start()

        output.serve_kernel_port_as_iframe(PORT, path=f"/{filename}", height=height, cache_in_notebook=True)

        PORT += 1

def display_features(model, sae, hook_point, feature_idx, tokens, output_filename):


    batch_size = 2048 
    minibatch_size_tokens = 128

    feature_vis_config_gpt = SaeVisConfig(
        hook_point=hook_point,
        features=feature_idx,
        batch_size=batch_size,
        minibatch_size_tokens=minibatch_size_tokens,
        verbose=True,
    )

    sae_vis_data_gpt = SaeVisData.create(
        encoder=sae,
        model=model, # type: ignore
        tokens=tokens,  # type: ignore
        cfg=feature_vis_config_gpt,
    )

    for feature in feature_idx[:1]:
        sae_vis_data_gpt.save_feature_centric_vis(output_filename, feature)
        