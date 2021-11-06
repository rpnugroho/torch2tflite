import os
import click
import torch
from typing import Tuple, Optional
from torch2tflite.converter import torch_to_tflite
from torch2tflite.utils import load_torchvision_model, load_model_from_file


@click.group()
def cli():
    pass


@click.command(name="file")
@click.argument("input")
@click.argument("output")
@click.option(
    "-fp",
    "--precision",
    default="float16",
    type=str,
    help="Quantization type of TFLite model i.e. 'float16'",
)
@click.option(
    "-i",
    "--img-size",
    default=(224, 224),
    type=(int, int),
    help="Size of input image i.e. 244 244",
)
@click.option(
    "-b",
    "--batch-size",
    default=32,
    type=int,
    help="Batchsize i.e. 32",
)
def file2tflite(
    input: str, output: str, precision: str, img_size: tuple, batch_size: int
):
    """Convert PyTorch model from file to TFLite. \n
    INPUT   : PyTorch model filepath. \n
    OUTPUT  : Output name of TFLite model. \n
    """
    dummy_input = torch.randn(batch_size, 3, img_size[0], img_size[1])
    model = load_model_from_file(input)
    torch_to_tflite(model, output, dummy_input, precision=precision)


@click.command(name="vision")
@click.argument("model_name")
@click.argument("output")
@click.option(
    "-p",
    "--pretrained",
    default=True,
    type=bool,
    help="If True, get torchvision model pre-trained on ImageNet",
)
@click.option(
    "-fp",
    "--precision",
    default="float16",
    type=str,
    help="Quantization type of TFLite model i.e. 'float16'",
)
@click.option(
    "-i",
    "--img-size",
    default=(224, 224),
    type=(int, int),
    help="Size of input image i.e. 244 244",
)
@click.option(
    "-b",
    "--batch-size",
    default=32,
    type=int,
    help="Batchsize i.e. 32",
)
def vision2tflite(
    model_name: str,
    output: str,
    pretrained: bool,
    precision: str,
    img_size: tuple,
    batch_size: int,
):
    """Convert torchvision model from hub to TFLite. \n
    MODEL NAME : Model name i.e. "resnet50" \n
    OUTPUT     : Output name of TFLite model. \n
    """
    dummy_input = torch.randn(batch_size, 3, img_size[0], img_size[1])
    model = load_torchvision_model(model_name, pretrained=pretrained)
    torch_to_tflite(model, output, dummy_input, precision=precision)


cli.add_command(file2tflite)
cli.add_command(vision2tflite)
