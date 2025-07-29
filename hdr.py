import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from keras.models import load_model

model = load_model("digit_model.h5")

class DigitRecognizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognizer")
        self.geometry("600x400")  # BIGGER GUI

        self.canvas = tk.Canvas(self, width=300, height=300, bg='white')
        self.canvas.grid(row=0, column=0, pady=10, padx=10, columnspan=3)

        self.label = tk.Label(self, text="Draw a digit...", font=("Helvetica", 24))
        self.label.grid(row=0, column=3, padx=20)

        self.button_recognize = tk.Button(self, text="Recognise", command=self.predict_digit, width=15)
        self.button_recognize.grid(row=1, column=0, pady=10)

        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas, width=15)
        self.button_clear.grid(row=1, column=1, pady=10)

        self.canvas.bind("<B1-Motion>", self.draw)

        self.image1 = Image.new("L", (300, 300), "white")
        self.draw_image = ImageDraw.Draw(self.image1)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
        self.draw_image.ellipse([x - r, y - r, x + r, y + r], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_image.rectangle([0, 0, 300, 300], fill="white")
        self.label.configure(text="Draw a digit...")

    def predict_digit(self):
        # Resize to 28x28
        resized = self.image1.resize((28, 28))
        inverted = ImageOps.invert(resized)
        img_array = np.array(inverted).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = int(np.max(prediction) * 100)

        self.label.configure(text=f"{digit}, {confidence}%")

if __name__ == "__main__":
    app = DigitRecognizer()
    app.mainloop()
