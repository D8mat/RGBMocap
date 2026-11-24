"""RGB MoCap Max transport. Uses the shared bone/reference and lossless Bake protocol."""
from .bone_stream import BoneStream
PORT = 49169

class MaxStream(BoneStream):
    def __init__(self, port=PORT):
        super().__init__(port)
