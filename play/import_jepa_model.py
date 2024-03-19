import jepa.models.vision_transformer as vit
import logging
import torch
import os
from types import MethodType

# config fields from the config file pretrain

from datetime import datetime

now = datetime.now()

logging.basicConfig(filename=f"logs/eval_{now.strftime('%m-%d-%H-%M')}.log")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


model_name = "vit_large"
checkpoint_key = "target_encoder"
clip_duration = None
frames_per_clip = 16
tubelet_size = 2
uniform_power = True
use_silu = False
tight_silu = False
use_sdpa = True
patch_size = 16
folder = "/export/home/yang/git/jepa/ckpts/test/vitl"
checkpoint = "vitl16.pth.tar"  # name of pretrained model file inside folder"
write_tag = "jepa"
device = "cuda:0"

pretrained_path = os.path.join(folder, checkpoint)


def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    # Video specific parameters
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key="target_encoder",
):
    encoder = vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )

    encoder.to(device)
    encoder = load_pretrained(
        encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key
    )
    return encoder


def load_pretrained(encoder, pretrained, checkpoint_key="target_encoder"):
    logger.info(f"Loading pretrained model from {pretrained}")
    checkpoint = torch.load(pretrained, map_location="cpu")
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint["encoder"]

    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {
        k.replace("backbone.", ""): v for k, v in pretrained_dict.items()
    }
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(
                f'key "{k}" is of different shape in model and loaded state dict'
            )
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f"loaded pretrained model with msg: {msg}")
    logger.info(
        f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}'
    )
    del checkpoint
    return encoder


encoder = init_model(
    crop_size=224,
    device=device,
    pretrained=pretrained_path,
    model_name=model_name,
    patch_size=patch_size,
    tubelet_size=tubelet_size,
    frames_per_clip=frames_per_clip,
)


@torch.no_grad()
def output_shape(self):
    print("output shape")
    pass


encoder.output_shape = MethodType(output_shape, encoder)

breakpoint()

print("finished")
