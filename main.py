from typing import Tuple
from IPython.display import display
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)


has_cuda = torch.cuda.is_available()
device = torch.device("cpu" if not has_cuda else "cuda")


def get_base_model():
    # Create base model.
    options = model_and_diffusion_defaults()
    options["inpaint"] = True
    options["use_fp16"] = has_cuda
    options["timestep_respacing"] = "100"  # use 100 diffusion steps for fast sampling
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()

    if has_cuda:
        model.convert_to_fp16()

    model.to(device)
    model.load_state_dict(load_checkpoint("base-inpaint", device))
    return model, options, diffusion


def get_upsampler_model():
    # Create upsampler model.
    options_up = model_and_diffusion_defaults_upsampler()
    options_up["inpaint"] = True
    options_up["use_fp16"] = has_cuda
    options_up[
        "timestep_respacing"
    ] = "fast27"  # use 27 diffusion steps for very fast sampling
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cuda:
        model_up.convert_to_fp16()
    model_up.to(device)
    model_up.load_state_dict(load_checkpoint("upsample-inpaint", device))
    return model_up, options_up, diffusion_up


def show_images(batch: torch.Tensor):
    """Display a batch of images inline."""
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    display(Image.fromarray(reshaped.numpy()))


def read_image(path: str, size: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    pil_img = Image.open(path).convert("RGB")
    pil_img = pil_img.resize((size, size), resample=Image.BICUBIC)
    img = np.array(pil_img)
    return torch.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1


def process_image_sizes(image_tensor):
    shorter_side_length = min(image_tensor.shape[-1], image_tensor.shape[-2])
    image_tensor_centered = transforms.CenterCrop(shorter_side_length)(image_tensor)
    image_tensor_centered_64 = transforms.Resize(64)(image_tensor_centered)
    image_tensor_centered_256 = transforms.Resize(256)(image_tensor_centered)
    return image_tensor_centered_64.unsqueeze(0), image_tensor_centered_256.unsqueeze(0)


def preprocess(filename):
    input_image = Image.open(filename)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = preprocess(input_image)
    input_batch_64, input_batch_256 = process_image_sizes(input_tensor)
    return input_batch_64, input_batch_256


def produce_mask(input_batch):
    "Downloading DeepLabv3 segmentation model..."
    model2 = torch.hub.load(
        "pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True
    )
    model2.eval()

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model2.to("cuda")

    with torch.no_grad():
        output = model2(input_batch)["out"][0]

    return (output.argmax(0) > 0).float()


# Create an classifier-free guidance sampling function
def create_image(overlay, prompt, guidance_scale, batch_size):

    print("Downloading the base model...")
    model, options, diffusion = get_base_model()

    # Create the text tokens to feed to the model.
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(tokens, options["text_ctx"])

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options["text_ctx"]
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=torch.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=torch.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=torch.bool,
            device=device,
        ),
        inpaint_image=overlay.repeat(full_batch_size, 1, 1, 1).to(device) * 2 - 1,
        inpaint_mask=(overlay > 0)[0]
        .float()
        .repeat(full_batch_size, 1, 1, 1)
        .to(device),
    )

    def denoised_fn(x_start):
        # Force the model to have the exact right x_start predictions
        # for the part of the image which is known.
        return (
            x_start * (1 - model_kwargs["inpaint_mask"])
            + model_kwargs["inpaint_image"] * model_kwargs["inpaint_mask"]
        )

    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    print("Starting the image generation process...")

    # Sample from the base model.
    model.del_cache()
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
        denoised_fn=denoised_fn,
    )[:batch_size]
    model.del_cache()

    return samples


def upsample_image(
    high_res_overlay, low_res_overlay, prompt, guidance_scale, upsample_temp, batch_size
):

    print("Downloading upsampler model...")
    model_up, options_up, diffusion_up = get_upsampler_model()

    tokens = model_up.tokenizer.encode(prompt)
    tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
        tokens, options_up["text_ctx"]
    )
    # Create the model conditioning dict.
    model_kwargs = dict(
        # Low-res image to upsample.
        low_res=((low_res_overlay + 1) * 127.5).round() / 127.5 - 1,
        # Text tokens
        tokens=torch.tensor([tokens] * batch_size, device=device),
        mask=torch.tensor([mask] * batch_size, dtype=torch.bool, device=device),
        inpaint_image=high_res_overlay.repeat(batch_size, 1, 1, 1).to(device) * 2 - 1,
        inpaint_mask=(high_res_overlay > 0)[0]
        .float()
        .repeat(batch_size, 1, 1, 1)
        .to(device),
    )

    def denoised_fn(x_start):
        # Force the model to have the exact right x_start predictions
        # for the part of the image which is known.
        return (
            x_start * (1 - model_kwargs["inpaint_mask"])
            + model_kwargs["inpaint_image"] * model_kwargs["inpaint_mask"]
        )

    print("Starting the upsampling process...")
    # Sample from the base model.
    model_up.del_cache()
    up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
    up_samples = diffusion_up.p_sample_loop(
        model_up,
        up_shape,
        noise=torch.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
        denoised_fn=denoised_fn,
    )[:batch_size]
    model_up.del_cache()

    return up_samples


def run(filename, prompt, guidance_scale, upsample_temp, batch_size):
    input_batch_64, input_batch_256 = preprocess(filename)
    m = produce_mask(input_batch_256)
    m_64 = transforms.Resize(64)(m.unsqueeze(0)).squeeze(0)
    input_image = Image.open(filename).convert("RGB")
    a = np.array(input_image)
    a = transforms.ToTensor()(a)

    # Generate image using GLIDE
    overlay_64 = m_64.cpu() * process_image_sizes(a)[0]
    samples = create_image(overlay_64.squeeze(0), prompt, guidance_scale, batch_size)

    # Upsampling
    overlay_256 = m.cpu() * process_image_sizes(a)[1]
    up_samples = upsample_image(
        overlay_256.squeeze(0),
        samples,
        prompt,
        guidance_scale,
        upsample_temp,
        batch_size,
    )

    # Show images
    show_images(up_samples)
