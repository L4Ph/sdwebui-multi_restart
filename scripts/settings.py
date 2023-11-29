import modules.scripts as scripts
import gradio as gr
import os

from modules import shared
from modules import script_callbacks


def on_ui_settings():
    section = ("sampler-params", "Sampler parameters")
    shared.opts.add_option(
        "restart_steps",
        shared.OptionInfo(
            6,
            label="restart_steps for Multi Restart",
            component=gr.Slider,
            component_args={"minimum": 0, "maximum": 100, "step": 1},
            section=section,
        ),
    )


script_callbacks.on_ui_settings(on_ui_settings)
