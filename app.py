import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. إعدادات صفحة الموقع
st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁")
st.title("🫁 تطبيق الكشف عن الالتهاب الرئوي (Pneumonia)")
st.write("قم برفع صورة أشعة سينية للصدر (X-ray) وسيقوم الذكاء الاصطناعي بتشخيصها.")

# 2. دالة لتحميل النموذج (استخدمنا cache لكي لا يحمل النموذج في كل مرة نرفع صورة)
@st.cache_resource
def load_model():
    # تأكد أن اسم الملف هنا يطابق اسم الملف الذي قمت بتحميله
model = tf.keras.models.load_model('pneumonia_pretrained_model.h5', compile=False)    return model

model = load_model()

# 3. دالة لمعالجة الصورة قبل إدخالها للنموذج
def process_image(image):
    # تحويل الصورة إلى RGB (لأن النموذج الجاهز MobileNetV2 يتطلب 3 قنوات ألوان)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # تغيير حجم الصورة لتصبح 224x224
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    # إضافة بعد جديد (Batch dimension)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalization (قسمة على 255)
    img_array = img_array / 255.0
    return img_array

# 4. واجهة رفع الصورة
uploaded_file = st.file_uploader("اختر صورة أشعة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة التي رفعها المستخدم
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة المرفوعة', use_column_width=True)
    
    st.write("جاري التحليل...")
    
    # 5. التوقع والنتيجة
    processed_image = process_image(image)
    prediction = model.predict(processed_image)[0][0] # النتيجة ستكون رقم بين 0 و 1
    
    # إظهار النتيجة بناءً على الرقم
    if prediction > 0.5:
        st.error(f"⚠️ النتيجة: المريض مصاب بالالتهاب الرئوي (Pneumonia). نسبة التأكد: {prediction*100:.2f}%")
    else:
        st.success(f"✅ النتيجة: الرئتان سليمتان (Normal). نسبة التأكد: {(1-prediction)*100:.2f}%")

st.markdown("---")
st.write("تم تطوير هذا المشروع كجزء من مشروع التخرج - 2024")
