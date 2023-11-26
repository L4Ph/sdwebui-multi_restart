import modules.scripts as scripts
import gradio as gr
import os

from modules import shared
from modules import script_callbacks

def on_ui_settings():
    section = ('sampler-params', "Sampler parameters")
    shared.opts.add_option(
        "restart_steps",
        shared.OptionInfo(
            6,  # default
            "Number of restart steps",
            gr.Number,
            {"interactive": True},
            section=section)
    )

script_callbacks.on_ui_settings(on_ui_settings)