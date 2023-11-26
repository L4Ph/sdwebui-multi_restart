import modules.scripts as scripts
import gradio as gr
import os

from modules import shared
from modules import script_callbacks

def on_ui_settings():
        section = ('sampler-params', "Sampler parameters")
        shared.opts.add_option(
            "restart_steps",
            shared.OptionInfo(6.0, "restart_steps for Multi Restart",
            gr.Slider, {"minimum": 0.0, "maximum": 100.0, "step": 1.0},
            infotext='restart_steps').info("Value of restart_steps for Multi Restart")
        )

script_callbacks.on_ui_settings(on_ui_settings)