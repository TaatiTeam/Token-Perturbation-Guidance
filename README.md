# Token-Perturbation-Guidance
Official implementation of "Token Perturbation Guidance for Diffusion Models"

![](./assets/uncond_generation.jpg)

## Get started

This is an example of a Python script:
```python
from pipeline_tpg import StableDiffusionXLTPGPipeline
pipe = StableDiffusionXLTPGPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
)

device = "cuda"
pipe = pipe.to(device)
prompts = [""]
seed = 0

generator = torch.Generator(device="cuda").manual_seed(seed)
output = pipe(
    prompts,
    guidance_scale=0.0,
    tpg_scale=3.0,
    generator=generator,
).images
```
