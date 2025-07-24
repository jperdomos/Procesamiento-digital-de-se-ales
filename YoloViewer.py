import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import torch
import numpy as np
import time



MODEL_PATH = "best.torchscript"
DEVICE = "cpu"
CLASS_NAMES = [
    'Aloe vera (Sábila)', 'Calendula officinalis (Calendula)',
    'Chamaemelum nobile (Manzanilla)', 'Dysphania ambrosioides (Paico)',
    'Eryngium foetidum (Cimarrón)', 'Erythroxylum coca (Coca)',
    'Mentha spicata (Hierbabuena)', 'Peumus boldus (Boldo)',
    'PlantasNoMedicinales', 'Ruta graveolens (Ruda)',
    'Valeriana officinalis (Valeriana)'
]


info_plantas = {
    'Aloe vera (Sábila)': {
        'nombre_comun': 'Sábila o Aloe Vera',
        'nombre_cientifico': 'Aloe vera',
        'familia': 'Asphodelaceae',
        'descripcion': 'Planta suculenta con hojas carnosas y alargadas que contienen un gel transparente en su interior, muy apreciado por sus propiedades medicinales y cosméticas.',
        'usos_medicinales': [
            'Tratamiento de quemaduras solares y heridas leves.',
            'Hidratación profunda de la piel y el cabello.',
            'Alivio del estreñimiento (consumo del jugo).',
            'Cuidado post-afeitado o post-depilación.'
        ],
        'propiedades': ['Cicatrizante', 'Antiinflamatorio', 'Hidratante', 'Laxante suave'],
        'preparacion': 'Uso tópico del gel extraído directamente de la hoja. Consumo del jugo procesado.',
        'precauciones': 'El consumo interno debe ser moderado. La capa amarilla bajo la corteza (aloína) puede ser irritante y laxante fuerte; se recomienda retirarla.'
    },
    'Calendula officinalis (Calendula)': {
        'nombre_comun': 'Caléndula',
        'nombre_cientifico': 'Calendula officinalis',
        'familia': 'Asteraceae',
        'descripcion': 'Planta herbácea con flores anaranjadas o amarillas brillantes, muy utilizada en medicina natural y cosmética por sus propiedades antiinflamatorias y cicatrizantes.',
        'usos_medicinales': [
            'Tratamiento de heridas, cortes y quemaduras leves.',
            'Alivio de irritaciones cutáneas y eccemas.',
            'Como enjuague bucal para encías inflamadas.'
        ],
        'propiedades': ['Antiinflamatoria', 'Cicatrizante', 'Antiséptica', 'Emoliente'],
        'preparacion': 'Infusión de las flores para uso tópico o enjuagues. Pomadas y cremas de extracto de caléndula.',
        'precauciones': 'Evitar en personas alérgicas a plantas de la familia Asteraceae. No ingerir durante el embarazo sin supervisión médica.'
    },
    'Chamaemelum nobile (Manzanilla)': {
        'nombre_comun': 'Manzanilla',
        'nombre_cientifico': 'Chamaemelum nobile',
        'familia': 'Asteraceae',
        'descripcion': 'Pequeña planta con flores similares a las margaritas, muy conocida por sus efectos calmantes y digestivos.',
        'usos_medicinales': [
            'Alivio de molestias gastrointestinales como gases y cólicos.',
            'Inductor del sueño y relajante.',
            'Reducción de la inflamación ocular y cutánea.'
        ],
        'propiedades': ['Digestiva', 'Calmante', 'Antiinflamatoria', 'Sedante suave'],
        'preparacion': 'Infusión de las flores. Compresas para los ojos o la piel.',
        'precauciones': 'Puede causar alergia en personas sensibles a las Asteraceae. Evitar su uso continuo en grandes cantidades.'
    },
    'Dysphania ambrosioides (Paico)': {
        'nombre_comun': 'Paico o Epazote',
        'nombre_cientifico': 'Dysphania ambrosioides',
        'familia': 'Amaranthaceae',
        'descripcion': 'Planta aromática de uso tradicional en América Latina, especialmente en gastronomía y como vermífugo (contra parásitos).',
        'usos_medicinales': [
            'Eliminación de parásitos intestinales (vermífugo).',
            'Alivio de problemas digestivos como gases y cólicos.',
            'Estimulación del apetito.'
        ],
        'propiedades': ['Antiparasitaria', 'Carminativa', 'Estimulante digestivo'],
        'preparacion': 'Infusión ligera de las hojas. Uso culinario en pequeñas cantidades.',
        'precauciones': 'Tóxica en dosis altas por presencia de ascaridol. No recomendada en embarazo ni en niños pequeños.'
    },
    'Eryngium foetidum (Cimarrón)': {
        'nombre_comun': 'Cimarrón o Culantro',
        'nombre_cientifico': 'Eryngium foetidum',
        'familia': 'Apiaceae',
        'descripcion': 'Hierba aromática con hojas largas y dentadas. Su olor y sabor son similares al cilantro, pero más intensos.',
        'usos_medicinales': [
            'Alivio de dolores estomacales y gases.',
            'Tratamiento de la fiebre y el resfriado.',
            'Estímulo del apetito.'
        ],
        'propiedades': ['Digestivo', 'Carminativo', 'Antiinflamatorio', 'Febrífugo'],
        'preparacion': 'Infusión de las hojas. También como condimento en la cocina.',
        'precauciones': 'Seguro en cantidades culinarias. Consultar en dosis medicinales.'
    },
    'Erythroxylum coca (Coca)': {
        'nombre_comun': 'Hoja de Coca',
        'nombre_cientifico': 'Erythroxylum coca',
        'familia': 'Erythroxylaceae',
        'descripcion': 'Arbusto andino cuyas hojas son usadas tradicionalmente por sus efectos estimulantes y medicinales.',
        'usos_medicinales': [
            'Alivio del mal de altura.',
            'Reducción de la fatiga y el apetito.',
            'Mejora de la digestión.'
        ],
        'propiedades': ['Energizante', 'Digestiva', 'Estimulante'],
        'preparacion': 'Masticado o infusión (mate de coca).',
        'precauciones': 'Uso tradicional legal en algunos países andinos. Prohibida en otros por su alcaloide base.'
    },
    'Mentha spicata (Hierbabuena)': {
        'nombre_comun': 'Hierbabuena',
        'nombre_cientifico': 'Mentha spicata',
        'familia': 'Lamiaceae',
        'descripcion': 'Planta aromática ampliamente utilizada por su agradable aroma y sabor, así como por sus beneficios digestivos.',
        'usos_medicinales': [
            'Alivio de indigestiones y cólicos.',
            'Frescura bucal.',
            'Descongestionante respiratorio.'
        ],
        'propiedades': ['Carminativa', 'Digestiva', 'Antiséptica', 'Antiespasmódica'],
        'preparacion': 'Infusión de hojas, masticación directa.',
        'precauciones': 'Evitar exceso en embarazo o lactancia.'
    },
    'Peumus boldus (Boldo)': {
        'nombre_comun': 'Boldo',
        'nombre_cientifico': 'Peumus boldus',
        'familia': 'Monimiaceae',
        'descripcion': 'Árbol o arbusto chileno cuyas hojas aromáticas son reconocidas por sus efectos en el sistema digestivo y hepático.',
        'usos_medicinales': [
            'Estimulación de la producción de bilis.',
            'Tratamiento de indigestión y dispepsia.',
            'Alivio de trastornos hepáticos leves.'
        ],
        'propiedades': ['Hepatoprotector', 'Colerético', 'Digestivo'],
        'preparacion': 'Infusión de las hojas secas.',
        'precauciones': 'No usar en embarazo, lactancia ni con enfermedades hepáticas graves. Evitar uso prolongado.'
    },
    'Ruta graveolens (Ruda)': {
        'nombre_comun': 'Ruda',
        'nombre_cientifico': 'Ruta graveolens',
        'familia': 'Rutaceae',
        'descripcion': 'Arbusto de olor fuerte tradicionalmente empleado en medicina popular y rituales.',
        'usos_medicinales': [
            'Regulación del ciclo menstrual.',
            'Alivio de cólicos y dolores espasmódicos.',
            'Sedante natural.'
        ],
        'propiedades': ['Emenagoga', 'Antiespasmódica', 'Sedante'],
        'preparacion': 'Infusión en pequeñas dosis.',
        'precauciones': 'Tóxica en dosis altas. Prohibida en embarazo. Puede irritar la piel con el sol.'
    },
    'Valeriana officinalis (Valeriana)': {
        'nombre_comun': 'Valeriana',
        'nombre_cientifico': 'Valeriana officinalis',
        'familia': 'Caprifoliaceae',
        'descripcion': 'Planta herbácea con raíces sedantes, popularmente usada para dormir y calmar los nervios.',
        'usos_medicinales': [
            'Tratamiento del insomnio.',
            'Reducción del estrés y ansiedad.',
            'Alivio de nerviosismo e irritabilidad.'
        ],
        'propiedades': ['Sedante', 'Ansiolítica', 'Relajante'],
        'preparacion': 'Infusión de raíz, cápsulas o extracto.',
        'precauciones': 'No usar con alcohol ni con otros sedantes. Puede causar somnolencia.'
    },
    'PlantasNoMedicinales': {
        'nombre_comun': 'No Medicinal',
        'nombre_cientifico': 'N/A',
        'familia': 'N/A',
        'descripcion': 'Categoría genérica para elementos sin identificación médica reconocida en el contexto del proyecto.',
        'usos_medicinales': ['Ninguno conocido.'],
        'propiedades': ['Ninguna.'],
        'preparacion': 'No aplica.',
        'precauciones': 'No se recomienda su uso.'
    }
}


def load_model(path, device="cpu"):
    model = torch.jit.load(path, map_location=device)
    model.eval()
    return model

def preprocess_frame(frame, size=(640, 640)):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, size)
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return img_tensor, img_rgb

def classify(model, tensor, device="cpu"):
    tensor = tensor.to(device)
    with torch.no_grad():
        t0 = time.time()
        logits = model(tensor)
        t1 = time.time()
    probs = torch.softmax(logits, dim=1)[0]
    idx = torch.argmax(probs).item()
    return CLASS_NAMES[idx], probs[idx].item(), (t1 - t0) * 1000

class CameraClassifierApp(tk.Tk):
    def __init__(self, model, device="cpu"):
        super().__init__()
        self.title("Green Machine - Clasificador de Plantas")
        self.geometry("1200x700")
        self.model = model
        self.device = device
        self.input_size = (640, 640)

        # Marco principal dividido horizontalmente
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # === Lado izquierdo: cámara ===
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.video_label = ttk.Label(self.left_frame)
        self.video_label.pack()

        self.pred_label = ttk.Label(self.left_frame, text="Iniciando...", font=("Arial", 14, "bold"))
        self.pred_label.pack(pady=10)

        # === Lado derecho: información de planta ===
        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.info_title = ttk.Label(self.right_frame, text="Información de la planta", font=("Arial", 16, "bold"))
        self.info_title.pack(anchor="w")

        self.text_info = tk.Text(self.right_frame, wrap=tk.WORD, font=("Arial", 12))
        self.text_info.pack(fill=tk.BOTH, expand=True)

        # Cámara
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.pred_label.config(text="❌ No se pudo abrir la cámara")
            return

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            self.pred_label.config(text="❌ Error capturando imagen")
            return

        tensor, rgb_img = preprocess_frame(frame, self.input_size)
        label, prob, time_ms = classify(self.model, tensor, self.device)

        # Mostrar imagen
        img_pil = Image.fromarray(rgb_img).resize((480, 480))
        tk_img = ImageTk.PhotoImage(img_pil)
        self.video_label.configure(image=tk_img)
        self.video_label.image = tk_img

        # Mostrar etiqueta
        self.pred_label.config(text=f"🌿 {label}\n📊 Confianza: {prob*100:.2f}%")

        # Mostrar información
        planta = info_plantas.get(label, info_plantas['PlantasNoMedicinales'])

        texto = f"""🌱 Nombre común: {planta['nombre_comun']}
🔬 Nombre científico: {planta['nombre_cientifico']}
🏷️ Familia: {planta['familia']}
📄 Descripción: {planta['descripcion']}

💊 Usos medicinales:
- {"\n- ".join(planta['usos_medicinales'])}

🧪 Propiedades:
- {"\n- ".join(planta['propiedades'])}

🍵 Preparación: {planta['preparacion']}
⚠️ Precauciones: {planta['precauciones']}
"""
        self.text_info.delete(1.0, tk.END)
        self.text_info.insert(tk.END, texto)

        self.after(1000, self.update_frame)

    def on_close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    model = load_model(MODEL_PATH, DEVICE)
    app = CameraClassifierApp(model, DEVICE)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()