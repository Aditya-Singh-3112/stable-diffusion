import gradio as gr
import keras_cv
from tensorflow import keras
keras.mixed_precision.set_global_policy("float32")
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512, jit_compile=True)
def generate(prompt):
    image = model.text_to_image(prompt, batch_size = 1)
    return image

demo = gr.Interface(generate, inputs = 'text', outputs = 'image')
demo.launch()