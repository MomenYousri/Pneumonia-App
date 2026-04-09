import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. إعدادات صفحة الموقع
st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁")
st.title("🫁 تطبيق الكشف عن الالتهاب الرئوي (Pneumonia)")
st.write("قم برفع صورة أشعة سينية للصدر (X-ray) وسيقوم الذكاء الاصطناعي بتشخيصها.")

# 2. تحميل النموذج
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('pneumonia_pretrained_model.h5', compile=False)
    return model

model = load_model()

# 3. دالة معالجة الصورة
def process_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# 4. واجهة رفع الصورة
uploaded_file = st.file_uploader("اختر صورة أشعة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة المرفوعة', use_column_width=True)
    
    st.write("جاري التحليل...")
    
    # التوقع
    processed_image = process_image(image)
    prediction = model.predict(processed_image)[0][0]
    
    # النتيجة
    if prediction > 0.5:
        st.error(f"⚠️ النتيجة: المريض مصاب بالالتهاب الرئوي (Pneumonia). نسبة التأكد: {prediction*100:.2f}%")
    else:
        st.success(f"✅ النتيجة: الرئتان سليمتان (Normal). نسبة التأكد: {(1-prediction)*100:.2f}%")

st.markdown("---")
st.write("تم تطوير هذا المشروع كجزء من مشروع التخرج - 2024")
