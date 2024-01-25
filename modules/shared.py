import sys

import gradio as gr

from modules import shared_cmd_options, shared_gradio_themes, options, shared_items, sd_models_types
from modules.paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir  # noqa: F401
from modules import util

cmd_opts = shared_cmd_options.cmd_opts
parser = shared_cmd_options.parser

batch_cond_uncond = True  # old field, unused now in favor of shared.opts.batch_cond_uncond
parallel_processing_allowed = True
styles_filename = cmd_opts.styles_file
config_filename = cmd_opts.ui_settings_file
hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}

demo = None

device = None

weight_load_location = None

if os.environ.get('IGNORE_CMD_ARGS_ERRORS', None) is None:
    cmd_opts = parser.parse_args()
else:
    cmd_opts, _ = parser.parse_known_args()


restricted_opts = {
    "samples_filename_pattern",
    "directories_filename_pattern",
    "outdir_samples",
    "outdir_txt2img_samples",
    "outdir_img2img_samples",
    "outdir_extras_samples",
    "outdir_grids",
    "outdir_txt2img_grids",
    "outdir_save",
    "outdir_init_images"
}

# https://huggingface.co/datasets/freddyaboulton/gradio-theme-subdomains/resolve/main/subdomains.json
gradio_hf_hub_themes = [
    "gradio/glass",
    "gradio/monochrome",
    "gradio/seafoam",
    "gradio/soft",
    "freddyaboulton/dracula_revamped",
    "gradio/dracula_test",
    "abidlabs/dracula_test",
    "abidlabs/pakistan",
    "dawood/microsoft_windows",
    "ysharma/steampunk"
]


cmd_opts.disable_extension_access = (cmd_opts.share or cmd_opts.listen or cmd_opts.server_name) and not cmd_opts.enable_insecure_extension_access

devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_esrgan, devices.device_codeformer = \
    (devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'esrgan', 'codeformer'])

devices.dtype = torch.float32 if cmd_opts.no_half else torch.float16
devices.dtype_vae = torch.float32 if cmd_opts.no_half or cmd_opts.no_half_vae else torch.float16

device = devices.device
weight_load_location = None if cmd_opts.lowram else "cpu"

batch_cond_uncond = cmd_opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
parallel_processing_allowed = not cmd_opts.lowvram and not cmd_opts.medvram
xformers_available = False 
config_filename = cmd_opts.ui_settings_file

hypernetworks = {}

loaded_hypernetworks = []

state = None

prompt_styles = None

interrogator = None

face_restorers = []

options_templates = None
opts = None
restricted_opts = None

sd_model: sd_models_types.WebuiSdModel = None

settings_components = None
"""assinged from ui.py, a mapping on setting names to gradio components repsponsible for those settings"""

tab_names = []

latent_upscale_default_mode = "Latent"
latent_upscale_modes = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}

sd_upscalers = []

clip_model = None

progress_print_out = sys.stdout

gradio_theme = gr.themes.Base()

total_tqdm = None

mem_mon = None

options_section = options.options_section
OptionInfo = options.OptionInfo
OptionHTML = options.OptionHTML

natural_sort_key = util.natural_sort_key
listfiles = util.listfiles
html_path = util.html_path
html = util.html
walk_files = util.walk_files
ldm_print = util.ldm_print

reload_gradio_theme = shared_gradio_themes.reload_gradio_theme

list_checkpoint_tiles = shared_items.list_checkpoint_tiles
refresh_checkpoints = shared_items.refresh_checkpoints
list_samplers = shared_items.list_samplers
reload_hypernetworks = shared_items.reload_hypernetworks
