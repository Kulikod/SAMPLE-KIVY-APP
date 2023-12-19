from kivy.lang import Builder
from kivymd.app import MDApp
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from plyer import filechooser

# Load the model
model = load_model("plant_model.h5", compile=False)

# Load the labels
class_names = open("plant_labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

KV = '''
AnchorLayout:
    anchor_x: 'center'
    anchor_y: 'center'

    BoxLayout:
        adaptive_size: True
        orientation: 'vertical'
        padding: '10dp'
        spacing: '10dp'
        size_hint: None, None
        size: "400dp", "400dp"

        MDCard:
            orientation: 'vertical'
            padding: '10dp'
            size_hint: None, None
            size: "400dp", "400dp"

            BoxLayout:
                Image:
                    id: image
                    source: 'plant.png'
                    halign:"center"
                MDLabel:
                    id: result
                    text: 'Scan a plant'
                    theme_text_color: "Secondary"
                    allow_stretch: True
                    keep_ratio: True
            	    font_style:"Caption"
                    font_size: "20sp"

            BoxLayout:
                orientation: 'vertical'
                size_hint: None, None
                size: "400dp", "50dp"
                pos_hint: {'center_x': 0.5}

            AnchorLayout:
                anchor_x: 'center'
                anchor_y: 'bottom'

                MDRaisedButton:
                    adaptive_size: True
                    text: "Scan a plant"
                    on_release: app.load_image()
'''

class MyApp(MDApp):
    def build(self):
        return Builder.load_string(KV)

    def load_image(self):
        # Open file chooser
        file_path = filechooser.open_file(title="Wybierz zdjęcie rośliny", filters=[("Image files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            # Replace this with the path to your image
            image = Image.open(file_path[0]).convert("RGB")

            # resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

            # turn the image into a numpy array
            image_array = np.asarray(image)

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # Predicts the model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Update label text
            self.root.ids.result.text = "Klasa: " + class_name[2:] + "\nWynik pewności: " + str(confidence_score)

            # Update image source
            self.root.ids.image.source = file_path[0]

if __name__ == "__main__":
    MyApp().run()
