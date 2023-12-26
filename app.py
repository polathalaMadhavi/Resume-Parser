import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import easyocr
import spacy
from spacy import displacy
from pdf2image import convert_from_bytes

reader = easyocr.Reader(['en'])   #initializes an EasyOCR reader object for the English language

#to make boxes fro text 
def draw_boxes(image, bounds, color='yellow', width=2):
    image = Image.fromarray(image)  # Convert NumPy array to PIL Image
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return np.array(image)

def main():
    st.title("RESUME PARSER")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Convert the uploaded PDF file to images
        pdf_images = convert_from_bytes(uploaded_file.read(), dpi=200)

        # Process the first page only (example, you can modify this logic)
        if st.session_state.get('processed_pages') is None:
            st.session_state['processed_pages'] = 0

        if st.session_state['processed_pages'] < len(pdf_images):
            image = pdf_images[st.session_state['processed_pages']]

            st.image(image, caption='PDF page', use_column_width=True)

            bounds = reader.readtext(np.array(image))

            # Display text with bounding boxes
            image_with_boxes = draw_boxes(np.array(image), bounds)
            st.image(image_with_boxes, caption='Text with bounding boxes', use_column_width=True)

            # Extract and display text
            text = ''
            for i in range(len(bounds)):
              text += bounds[i][1] + '\n'+'\n'
            st.write("Extracted text:",text)
            # st.write(text)

            nlp = spacy.load('en_core_web_sm')
            doc = nlp(text)

            # Display SpaCy visualization
            # st.write("Skills Highlighted:")
            html = displacy.render(doc, style='ent', options={'distance': 120}, page=True)
            st.components.v1.html(html, width=800, height=600)
 # Perform Named Entity Recognition (NER) with SpaCy
      
            st.session_state['processed_pages'] += 1

if __name__ == "__main__":
    main()
