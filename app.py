import streamlit as st
import joblib
from PIL import Image
import glob

from facenet_pytorch import MTCNN, InceptionResnetV1, prewhiten
from torchvision import transforms
from face_recognition import preprocessing
import torch
from facenet_pytorch.models.utils.detect_face import extract_face

aligner = MTCNN(keep_all=True, thresholds=[0.9, 0.9, 0.9])
facenet = InceptionResnetV1(pretrained='vggface2').eval()
facenet_preprocess = transforms.Compose([preprocessing.Whitening()])

from collections import namedtuple
Prediction = namedtuple('Prediction', 'label confidence')

def top_prediction(idx_to_class, probs):
    top_label = probs.argmax()
    return Prediction(label=idx_to_class[top_label], confidence=probs[top_label])


@st.cache(allow_output_mutation=True)
def load_model(model_name):
    face_recogniser = joblib.load(model_name)
    return face_recogniser


def main():
    """
        Face Matching
    """
    
    activity = ["CELEB MATCH"]
    choice = st.sidebar.selectbox("Choose Activity",activity)
    
    #CELEB MATCH
    if choice == "CELEB MATCH":
        face_recogniser = load_model('model/face_recogniser.pkl')
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = image.convert("RGB")
            bbs, _ = aligner.detect(image)
            if bbs is not None:
                faces = torch.stack([extract_face(image, bb) for bb in bbs])
                embeddings = facenet(facenet_preprocess(faces)).detach().numpy()
                predictions = face_recogniser.classifier.predict_proba(embeddings)
                for bb, probs in zip(bbs, predictions):
                    try:
                        cropped_faces = []
                        cropped_face = image.crop(bb)
                        cropped_faces.append(cropped_face)
                        prediction = top_prediction(face_recogniser.idx_to_class, probs)
                        files = glob.glob("images/" + prediction.label + "/*.*")
                        actor_image = Image.open(files[0])
                        actor_image_bbs, _ = aligner.detect(actor_image)
                        actor_image = actor_image.crop(actor_image_bbs[0]) if len(actor_image_bbs) > 0 else actor_image
                        cropped_faces.append(actor_image)
                        st.image(cropped_faces, width=100)
                        st.write(prediction.label)
                    except:
                        pass
            else:
                st.write("Can't detect face")
            st.image(image, caption='Uploaded Image.', use_column_width=True)
    
if __name__ == "__main__":
    main()