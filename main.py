import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from edit_pic_with_ai import detect_and_segment, cropped_image
from stabledif import PictureGen

app = FastAPI()

@app.post(f"/create_picture_and_cropping/")
async def create_picture(description: str):
    pic_gen = PictureGen(description)
    file_name = pic_gen.save_pic()
    coords = detect_and_segment(file_name)
    new_img = cropped_image(file_name, coords)
    file_res = FileResponse(new_img)
    return file_res


def main() -> None:
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()