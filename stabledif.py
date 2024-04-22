import uuid
from diffusers import DiffusionPipeline



class PictureGen:
    def __init__(self, text):
        self.pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.text = text
        self.image = self.pipeline(text).images[0]

    def save_pic(self):
        file_name = f"{uuid.uuid4()}.png"
        self.image.save(file_name)
        return file_name