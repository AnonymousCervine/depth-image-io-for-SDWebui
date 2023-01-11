# Re: Copyright and licenses
#
# Firstly: This uses a few lines of boilerplate code from the Stable Diffusion 2.0 reference code at https://github.com/Stability-AI/stablediffusion
# To the extent those few lines of boilerplate are copyrightable, the MIT license applies as per https://github.com/Stability-AI/stablediffusion/blob/main/LICENSE
#
# Secondly: This uses some lines of code from https://github.com/AUTOMATIC1111/stable-diffusion-webui, with which this code is designed to interoperate; almost all of the
# code in question from this latter source is, in turn, lightly-modified boilerplate from the former.
# stable-diffusion-webui, however, has no license. The best that can be said is that de-facto it is generally understood that permission is granted to copy the code for (at least)
# personal use by its (myriad and various) contributors.
#
# Because of that quibble I cannot in good faith assign a license to this code; it is not fully clear to me that I have the rights to do so in the first place (even if
# I BELEIVE in good faith that the few lines reproduced here are very likely not-sufficient in nature, nor extent, to constitute any copyright violations via their inclusion).
# Notwithstanding that, for all such portions of the code as I do own, I freely permit the use of them as-is, with no representations made as to their performance nor dangers.
# ...That much should, I think, be enough for anyone who is willing, in the first place, to use the nebulously-copyrighted codebase that this code is designed to operate with.

from types import MethodType
import gradio as gr
import numpy as np
from PIL import Image
import torch
from einops import repeat, rearrange
from ldm.data.util import AddMiDaS

import modules.processing as processing
import modules.scripts as scripts
import modules.shared as shared

instructions = """\
### Depth Image I/O
###### (Only applicable to Depth2Image models)

This is a script for the combined purposes of: **A)** Inserting custom depth images *into* a Depth2Img model and **B)** Getting the depth images *generated* by MiDaS back out (when a custom depth image is *not* specified). The depth2img model can infer quite a lot from just depth!

**General Notes and Observations:**
 - The depth image should be greyscale (Accordingly: Anything beyond the first channel (i.e. red) of any RGB image will be ignored; if you put a color image in, it will gamely try to interpret the red channel as depth with probably undesired results!)
 - White is nearest to the camera, black is farthest
   - If you don't use the whole black-to-white range, the image will be normalized to the full range automatically (which is what the model expects and was trained on, although in reality it isn't too picky).
 - The distance values should theoretically be linear (although again, in practice, it turns out it's not terribly picky about this)
   - But pick your range wisely, and 'faraway' distances should all perhaps just be sort of uniformly blackish, depending(see next bullet).
 - **FOR BEST RESULTS with handmade input, base your inputs on what the auto-generated depth-images look like.**
   - All the important features of your composition should take up most-if-not-all of the available depth space
   - To permit that, anything in the "distant (or even not-so-distant) background" can just be a uniform-ish black/dark-grey
     - Such "far from the camera" distances seem indeed to be treated by this model as "draw whatever you like past here"
   - Alternatively, if one wants to keep faraway details, one may instead make them smaller and closer than they physically would be (in order to fit nicely into the limited depth space).
  - **NOTE**: You can leave `Append adjusted depth image to outputs` checked when using your own images, to confirm that your depth values look the way you expect after being normalized.
 - Note that the image will be downscaled down to 1/8th of the target image size (so 64x64 for 512x512 output) internally, so fine details may be lost.
   - That said the model can extrapolate a surprising amount of detail from a downscaled 64x64 image!

*Finally, be aware that this code is slightly fragile and may break in a future update! Have fun, and good luck!*
"""

class Script(scripts.Script):
    def title(self):
        return "Custom Depth Images (input/output)"

    def ui(self, is_img2img):
        show_depth = gr.Checkbox(True, label="Append (normalized) depth image to outputs\n(yours if supplied; the auto-generated if otherwise)")
        with gr.Accordion("Notes and Hints (click to expand)", open=False):
            gr.Markdown(instructions)
        gr.Markdown("---\n\nPut depth image here ⤵")
        input_depth_img = gr.Image(source='upload', type="pil")
        return [input_depth_img, show_depth]

    def run(self, p, input_depth_img, show_depth):
        is_img2img = p is processing.StableDiffusionProcessingImg2Img
        use_custom_depth_input = bool(input_depth_img)
        if not is_img2img and not use_custom_depth_input:
            raise RuntimeError("If using the Custom Depth Images I/O script with txt2img, you MUST PROVIDE A DEPTH IMAGE (because there's no other image to infer depth from!)")
        p.out_depth_image = None
    
        # Monkeypatch depth2img_image_conditioning on this Processing instance to let us feed in our own depth and/or capture that which is autogenerated.
        old_depth2img_image_conditioning = p.depth2img_image_conditioning
        def alt_depth_image_conditioning(self, source_image):
            conditioning_image = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(source_image))
            if use_custom_depth_input:
                depth_data = rearrange(np.array(input_depth_img.convert("RGB")), "h w c -> c 1 1 h w")[0] # Rearrange and discard anything past the first channel.
                                                                                                        # Curious minds may want to know why we convert to 'RGB'; this is because it's 
                                                                                                        # theoretically possible the image was for instance palletized. Converting to 'L' mode 
                                                                                                        # is also possible but the image is LIKELY to in fact be a black-and-white RGB image
                                                                                                        # (because almost no one uses single-channel images) so this is likely to be more efficient.
                depth_data = torch.from_numpy(depth_data).to(device=shared.device).to(dtype=torch.float32) # That the range is 1.0-255.0 doesn't matter; we're going to normalize it anyway.
                depth_data = repeat(depth_data, "1 ... -> n ...", n=self.batch_size)
            else:
                transformer = AddMiDaS(model_type="dpt_hybrid")
                transformed = transformer({"jpg": rearrange(source_image[0], "c h w -> h w c")})
                midas_in = torch.from_numpy(transformed["midas_in"][None, ...]).to(device=shared.device)
                midas_in = repeat(midas_in, "1 ... -> n ...", n=self.batch_size)
                depth_data = self.sd_model.depth_model(midas_in)

            if show_depth:
                (depth_min, depth_max) = torch.aminmax(depth_data)
                display_depth = (depth_data - depth_min) / (depth_max - depth_min)
                depth_image = Image.fromarray(
                        (display_depth[0, 0, ...].cpu().numpy() * 255.).astype(np.uint8))
                self.out_depth_image = depth_image
            
            conditioning = torch.nn.functional.interpolate(
                depth_data,
                size=conditioning_image.shape[2:],
                mode="bicubic",
                align_corners=False,
            )
            (depth_min, depth_max) = torch.aminmax(conditioning) #recalculating may be unneccessary but bicubic interpolation would theoretically allow these to have expanded infinitesimally.
            conditioning = 2. * (conditioning - depth_min) / (depth_max - depth_min) - 1.
            return conditioning
        p.depth2img_image_conditioning = MethodType(alt_depth_image_conditioning, p)
        
        # Also monkeypatch txt2img_image_conditioning to handle txt2img side
        # This approach is just barely possible solely because the txt2img_image_conditioning method already exists to fill in a blank mask on dedicated inpainting models.
        def alt_txt2img_image_conditioning(self, x, width=None, height=None):
            fake_img = torch.zeros(1, 3, height or self.height, width or self.width).to(shared.device).type(self.sd_model.dtype) # Single fake input rgb 'image', used just to conform to existing code paths.
                                                                                                                                 # Ultimately all we'll get from processing it is learning that the target depth
                                                                                                                                 # image size is (width/8, depth/8)
            return self.depth2img_image_conditioning(fake_img) # Point at our new depth2img_image_conditioning function that we've just overridden.
        p.txt2img_image_conditioning = MethodType(alt_txt2img_image_conditioning, p)

        processed_output = processing.process_images(p)
        if show_depth and p.out_depth_image:
            processed_output.images.append(p.out_depth_image)
        return processed_output