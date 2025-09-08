import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import io
import base64
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import ultralytics
from ultralytics import YOLO


# ----------------------------
# Configuración inicial
# ----------------------------
def init_session_state():
    """Inicializar variables de sesión"""
    if "capturar" not in st.session_state:
        st.session_state.capturar = False
    if "imagen_actual" not in st.session_state:
        st.session_state.imagen_actual = None
    if "detecciones_historial" not in st.session_state:
        st.session_state.detecciones_historial = []
    if "modelo_cargado" not in st.session_state:
        st.session_state.modelo_cargado = False

def configurar_pagina():
    """Configurar la página de Streamlit"""
    st.set_page_config(
        page_title="🍎 RED NEURONAL CNN - Detección de Frutas",
        page_icon="🍎",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# ----------------------------
# Funciones de utilidad
# ----------------------------
@st.cache_data
def cargar_modelo():
    """Cargar el modelo CNN preentrenado (simulado aquí)"""
    ruta_modelo = r'C:\Users\USUARIO\Desktop\RED_NEURONAL\runs\detect\exp\weights\best.pt'
    model = YOLO(ruta_modelo)
    st.session_state.modelo_cargado = True
    return model


    """def cargar_clases_frutas():
    return [
        "Manzana", "Naranja", "Banana", "Mango", "Fresa", 
        "Uva", "Piña", "Pera", "Kiwi", "Limón"
    ]"""

    """def simular_modelo_cnn(imagen, confianza_min=0.5):
    
    clases = cargar_clases_frutas()
    
    # Simulación de detecciones aleatorias pero realistas
    np.random.seed(int(time.time()) % 100)
    num_detecciones = np.random.randint(1, 4)
    
    detecciones = []
    for i in range(num_detecciones):
        clase = np.random.choice(clases)
        confianza = np.random.uniform(confianza_min, 0.98)
        bbox = {
            'x1': np.random.randint(50, 200),
            'y1': np.random.randint(50, 200),
            'x2': np.random.randint(300, 500),
            'y2': np.random.randint(300, 500)
        }
        
        detecciones.append({
            "clase": clase,
            "confianza": round(confianza, 3),
            "bbox": bbox,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    return sorted(detecciones, key=lambda x: x['confianza'], reverse=True)"""
def procesar_imagen_con_modelo(model, imagen, confianza_min=0.5):
    modelo = cargar_modelo()
    resultados = modelo.predict(
        source=imagen, 
        conf=confianza_min, 
        imgsz=640,
        verbose=False)

        # Imagen con las detecciones dibujadas
    #img_resultado = resultados[0].plot()  

    # Lista con las detecciones
    detecciones = []
    img_resultado = None
    for r in resultados:
        img_resultado = r.plot()
        for box in r.boxes:
            clase_id = int(box.cls[0].item())
            clase = modelo.names[clase_id]  # nombre de la clase
            confianza = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detecciones.append({
                "clase": clase,
                "confianza": round(confianza, 3),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

    return img_resultado, sorted(detecciones, key=lambda x: x['confianza'], reverse=True)
        
def graficar_cantidad_por_clase(detecciones):
    """
    Genera un gráfico de barras mostrando cuántas frutas se detectaron por clase
    """
    if not detecciones:
        return None

    # Contar cuántas veces aparece cada clase
    clases = [d['clase'] for d in detecciones]
    df = pd.DataFrame({'clase': clases})
    conteo = df.value_counts().reset_index(name='cantidad')
    conteo.columns = ['clase', 'cantidad']

    # Crear gráfico
    fig = px.bar(
        conteo,
        x='clase',
        y='cantidad',
        title="Cantidad de frutas detectadas por clase",
        text='cantidad',
        color='clase',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Clase de fruta",
        yaxis_title="Cantidad",
        yaxis=dict(tick0=0, dtick=1)
    )
    return fig

def dibujar_detecciones(imagen, detecciones):
    """Dibujar bounding boxes en la imagen"""
    img_con_boxes = imagen.copy()
    
    for det in detecciones:
        bbox = det['bbox']
        cv2.rectangle(
            img_con_boxes,
            (bbox['x1'], bbox['y1']),
            (bbox['x2'], bbox['y2']),
            (0, 255, 0), 4 # color verde tamaño de linea 4
        )
        
        # Etiqueta
        label = f"{det['clase']}: {det['confianza']:.2f}"
        cv2.putText(
            img_con_boxes,
            label,
            (bbox['x1'], bbox['y1'] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    
    return img_con_boxes

def listar_cantidad_por_clase(detecciones):
    if not detecciones:
        return {}
    clases = [d['clase'] for d in detecciones]
    conteo = {}
    for clase in clases:
        conteo[clase] = conteo.get(clase, 0) + 1
    return conteo


def crear_grafico_confianza(detecciones):
    """Crear gráfico de barras con las confianzas"""
    if not detecciones:
        return None
    
    df = pd.DataFrame(detecciones)
    
    fig = px.bar(
        df, 
        x='clase', 
        y='confianza',
        title="Nivel de Confianza por Fruta Detectada",
        color='confianza',
        color_continuous_scale='Viridis',
        text='confianza'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        xaxis_title="Tipo de Fruta",
        yaxis_title="Confianza",
        yaxis=dict(range=[0, 1])
    )
    
    return fig

# ----------------------------
# Interfaz principal
# ----------------------------
def main():
    configurar_pagina()
    init_session_state()
    
    # Título y descripción
    st.title("🍎 *** CNN - Detección de estado de fruta")
    st.markdown("""
    ### 🚀 Sistema inteligente de reconocimiento de frutas
    Utiliza inteligencia artificial para detectar y clasificar el estado de la fruta en tiempo real.
    """)
    
    # Sidebar con controles
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Método de entrada
        metodo_entrada = st.radio(
            "Selecciona el método de entrada:",
            ["📷 tiempo real", "📁 Subir imagen", "🎯 Imagen de ejemplo"]
        )
        
        # Parámetros del modelo
        st.subheader("🔧 Parámetros del modelo")
        confianza_min = st.slider("Umbral mínimo de confianza", 0.0, 1.0, 0.5, 0.01)
        mostrar_bbox = st.checkbox("Mostrar cajas delimitadoras", True)
        
        # Información del sistema
        st.subheader("📊 Estado del sistema")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Modelo", "✅ Activo" if st.session_state.modelo_cargado else "⏳ Cargando")
        with col2:
            st.metric("Detecciones", len(st.session_state.detecciones_historial))
    
    # Layout principal
    col_main, col_results = st.columns([2, 1])
    
    with col_main:
        st.header("📸 Captura y Procesamiento")
        
        # Diferentes métodos de entrada
        imagen_procesada = None
        
        if metodo_entrada == "📷 Cámara en vivo":
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button("📸 Capturar Imagen", type="primary", use_container_width=True):
                    with st.spinner("Accediendo a la cámara..."):
                        try:
                            cap = cv2.VideoCapture(0)
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            
                            ret, frame = cap.read()
                            cap.release()
                            
                            if ret:
                                st.session_state.imagen_actual = frame
                                st.success("✅ Imagen capturada correctamente")
                            else:
                                st.error("❌ No se pudo acceder a la cámara")
                        except Exception as e:
                            st.error(f"❌ Error de cámara: {str(e)}")
            
        elif metodo_entrada == "📁 Subir imagen":
            archivo_subido = st.file_uploader(
                "Selecciona una imagen",
                type=['jpg', 'jpeg', 'png'],
                help="Formatos soportados: JPG, JPEG, PNG"
            )
            
            if archivo_subido is not None:
                imagen_pil = Image.open(archivo_subido)
                st.session_state.imagen_actual = cv2.cvtColor(
                    np.array(imagen_pil), cv2.COLOR_RGB2BGR
                )
        
        elif metodo_entrada == "🎯 Imagen de ejemplo":
            ejemplos = {
                "Manzanas rojas": "🍎",
                "Cítricos variados": "🍊", 
                "Frutas tropicales": "🥭"
            }
            
            ejemplo_seleccionado = st.selectbox(
                "Selecciona un ejemplo:",
                list(ejemplos.keys())
            )
            
            if st.button("Cargar ejemplo", type="secondary"):
                # Crear imagen de ejemplo (placeholder)
                img_ejemplo = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                st.session_state.imagen_actual = img_ejemplo
                st.info(f"📸 Cargado: {ejemplo_seleccionado} {ejemplos[ejemplo_seleccionado]}")
        
        # Mostrar imagen actual
        if st.session_state.imagen_actual is not None:
            st.subheader("🖼️ Imagen a procesar")
            st.image(st.session_state.imagen_actual, channels="BGR", use_column_width=True)
            
            # Botón de procesamiento
            if st.button("🔍 Procesar con CNN", type="primary", use_container_width=True):
                with st.spinner("🧠 Procesando con red neuronal..."):
                    time.sleep(1)  # Simular tiempo de procesamiento
                    
                    # Ejecutar modelo
                    model = cargar_modelo()
                    img_resultado, detecciones =procesar_imagen_con_modelo(
                        model,
                        st.session_state.imagen_actual, 
                        confianza_min
                    )
                    
                    # Guardar en historial
                    st.session_state.detecciones_historial.extend(detecciones)
                    
                    # Mostrar resultado
                    if detecciones:
                        st.subheader("🎯 Resultado con detecciones")
                        st.image(img_resultado, channels="BGR", use_column_width=True)
                        st.success(f"✅ Se detectaron {len(detecciones)} frutas")

                        #Mostrar gráfico de cantidad por clase
                        st.subheader("📊 Cantidad de frutas por clase")
                        fig_cantidad = graficar_cantidad_por_clase(detecciones)
                        if fig_cantidad:
                            st.plotly_chart(fig_cantidad, use_container_width=True)

                        # Mostrar lista
                        st.subheader("📝 Lista de frutas detectadas")
                        lista_cantidades = listar_cantidad_por_clase(detecciones)
                        for fruta, cantidad in lista_cantidades.items():
                            st.write(f"- {fruta}: {cantidad}")
                            
                   
    with col_results:
        st.header("📈 Resultados")
        
        # Mostrar última detección
        if st.session_state.detecciones_historial:
            ultimas_detecciones = st.session_state.detecciones_historial[-3:]
            
            st.subheader("🎯 Última detección")
            df_detecciones = pd.DataFrame(ultimas_detecciones)
            st.dataframe(
                df_detecciones[['clase', 'confianza', 'timestamp']],
                use_container_width=True,
                hide_index=True
            )
            
            # Gráfico de confianza
            st.subheader("📊 Análisis de confianza")
            fig = crear_grafico_confianza(ultimas_detecciones)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Estadísticas
            st.subheader("📈 Estadísticas")
            frutas_detectadas = [d['clase'] for d in st.session_state.detecciones_historial]
            fruta_mas_comun = max(set(frutas_detectadas), key=frutas_detectadas.count) if frutas_detectadas else "N/A"
            confianza_promedio = np.mean([d['confianza'] for d in st.session_state.detecciones_historial]) if st.session_state.detecciones_historial else 0
            
            col1, col2= st.columns(2)
            with col1:
                st.metric("Fruta más detectada", fruta_mas_comun)
            with col2:
                st.metric("Confianza promedio", f"{confianza_promedio:.2f}")
        
        else:
            st.info("👆 Procesa una imagen para ver los resultados aquí")
    
    # Footer con información adicional
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔧 Tecnología:**")
        st.markdown("- OpenCV + CNN Real/Local")
        st.markdown("- TensorFlow/PyTorch/ONNX")
        st.markdown("- Streamlit Dashboard")
        st.markdown("- Procesamiento en tiempo real")
    
    with col2:
        st.markdown("**🎯 Características:**")
        st.markdown("- Modelo CNN personalizado")
        st.markdown("- Carga automática de modelos")
        st.markdown("- Análisis de confianza")
        st.markdown("- Historial de detecciones")
    
    with col3:
        col_buttons = st.columns(2)
        with col_buttons[0]:
            if st.button("🗑️ Limpiar historial"):
                st.session_state.detecciones_historial = []
                st.success("✅ Historial limpiado")
        with col_buttons[1]:
            if st.button("💾 Exportar datos"):
                if st.session_state.detecciones_historial:
                    df_export = pd.DataFrame(st.session_state.detecciones_historial)
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar CSV",
                        data=csv,
                        file_name=f"detecciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No hay datos para exportar")

if __name__ == "__main__":
    main()