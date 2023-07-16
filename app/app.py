from fastai.vision.all import *
from fastai.vision.all import load_learner
import gradio as gr

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

Wildflower_species = (
    'Alpine Aster',
    'Alpine Forget-me-not',
    'Beach Evening Primrose',
    'Beach Morning Glory',
    'Black-eyed Susan',
    'Bluebell',
    'Columbine',
    'Daisy',
    'Desert Marigold',
    'Desert Sunflower',
    'Indian Paintbrush',
    "Lady's Slipper Orchid",
    'Lupine', 'Ocotillo',
    'Pitcher Plant',
    'Prairie Clover',
    'Prairie Phlox',
    'Prickly Pear',
    "Queen Anne's Lace",
    'Seaside Goldenrod',
    'Sunflower',
    'Swamp Milkweed',
    'Trillium',
    'Water Lily',
    'Wild Bergamot'
)

model = load_learner('models/Wildflower-recognizer-v2.pkl')


def recognize_image(image):
    pred, idx, probs = model.predict(image)
    return dict(zip(Wildflower_species, map(float, probs)))


image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label(num_top_classes=5)
examples = [
    'test_images/unknown_00.jpg',
    'test_images/unknown_01.jpg',
    'test_images/unknown_02.jpg',
    'test_images/unknown_04.jpg',
    'test_images/unknown_05.jpg',
    'test_images/unknown_06.jpg',
    'test_images/unknown_07.jpg',
    'test_images/unknown_08.jpg',
    'test_images/unknown_09.jpg',
    'test_images/unknown_10.jpg'
]

iface = gr.Interface(fn=recognize_image, inputs=image,
                     outputs=label, examples=examples)
iface.launch(inline=False, share=True)
