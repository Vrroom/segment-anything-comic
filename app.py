import os
import cv2
import numpy as np
import gradio as gr
from model import *


# points color and marker
color = (0, 255, 0)
marker = 5
model = load_model('lightning_logs/version_2')


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# SAM x Comics")

    with gr.Row().style(equal_height=True):
        with gr.Column():
            # input image
            original_image = gr.State(value=None)   # store original image without points, default None
            input_image = gr.Image(type="numpy")
            # point prompt
            with gr.Column():
                selected_points = gr.State([])      # store points
                output_points = gr.State([])        # store outputs
                with gr.Row():
                    # gr.Markdown('You can click on the image to select points prompt. Default: foreground_point.')
                    undo_button = gr.Button('Undo point')
            # run button
            button = gr.Button("Run")
        # show the image with result
        with gr.Tab(label='Result'):
            output_image = gr.Image(type='numpy')

    # once user upload an image, the original image is stored in `original_image`
    def store_img(img):
        return img, []  # when new image is uploaded, `selected_points` should be empty

    input_image.upload(
        store_img,
        [input_image],
        [original_image, selected_points]
    )

    # user click the image to get points, and show the points on the image
    def get_point(img, sel_pix, evt: gr.SelectData):
        sel_pix.append(evt.index)   # append the foreground_point
        for point in sel_pix:
            cv2.drawMarker(img, point, color, markerType=marker, markerSize=20, thickness=5)
        return img if isinstance(img, np.ndarray) else np.array(img)

    input_image.select(
        get_point,
        [input_image, selected_points], 
        [input_image],
    )

    # undo the selected point
    def undo_points(orig_img, sel_pix):
        temp = orig_img.copy()
        # draw points
        if len(sel_pix) != 0:
            sel_pix.pop()
            for point, label in sel_pix:
                cv2.drawMarker(temp, point, color, markerType=marker, markerSize=20, thickness=5)
        return temp if isinstance(temp, np.ndarray) else np.array(temp)

    undo_button.click(
        undo_points,
        [original_image, selected_points],
        [input_image]
    )

    # button image
    button.click(model.run_inference, inputs=[original_image, selected_points],
                 outputs=[output_image, output_points])

demo.queue().launch(debug=True, enable_queue=True, server_name='0.0.0.0')
