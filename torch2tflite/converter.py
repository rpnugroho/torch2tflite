import os
import tempfile
import torch
import onnx
import onnx_tf
import tensorflow as tf
from torch2tflite.utils import get_filename


def torch_to_onnx(model, output, dummy_input):
    """Convert pytorch model to onnx.
    https://pytorch.org/docs/stable/onnx.html

    Args:
        model ([type]): PyTorch model
        output (str): Output filename
        dummy_input (tuple or tensor): Dummy input
    """
    input_names = ["actual_input"]
    output_names = ["output"]
    print("Converting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )
    print(f"ONNX exported to {output}")


def onnx_to_tf(input: str, output: str):
    """Convert ONNX file to TensorFlow SavedModel

    Args:
        input (str): ONNX filename
        output (str): Directory of SavedModel
    """
    onnx_model = onnx.load(input)
    tf_model = onnx_tf.backend.prepare(onnx_model)
    print("Converting to TFSavedModel...")
    tf_model.export_graph(output)
    print(f"TensorFlow savedmodel saved to {output}")


def tf_to_tflite(input: str, output: str, precision: str = "float16"):
    """Convert TensorFLow SavedModel to TFLite

    Args:
        input (str): Directory of SavedModel
        output (str): Output filename
        precision (str, optional): Quantization type of TFLite model. Defaults to "float16".
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(input)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [getattr(tf, precision)]
    print("Converting to TFLite...")
    tflite_rep = converter.convert()
    open(output, "wb").write(tflite_rep)
    print(f"TFlite model saved to {output}")


def torch_to_tflite(model, output: str, dummy_input: str, precision: str = "float16"):
    """Convert PyTorch model to TFLite

    Args:
        model ([type]): PyTorch model.
        output (str): Output filename
        dummy_input (tuple or tensor): Dummy input
        precision (str, optional): Quantization type of TFLite model. Defaults to "float16".
    """
    filename = get_filename(output)
    # create temp dir
    print("Create temp directory")
    temp_dir = tempfile.TemporaryDirectory()
    temp_onnx = os.path.join(temp_dir.name, filename + ".onnx")
    temp_tf = os.path.join(temp_dir.name, filename)
    # convert to onnx
    torch_to_onnx(model=model, output=temp_onnx, dummy_input=dummy_input)
    # convert to tf
    onnx_to_tf(input=temp_onnx, output=temp_tf)
    # convert to tflite
    tf_to_tflite(input=temp_tf, output=output, precision=precision)
    # clean temp dir
    print("Clean temp directory")
    temp_dir.cleanup()
