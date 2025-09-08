import gdown
import os
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import io
import base64
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import ultralytics
from ultralytics import YOLO

ruta_modelo = "best.pt"
if not os.path.exists(ruta_modelo):
    url = "https://drive.google.com/file/d/1SFGVrcUS4DUPeVdBDrAAsU1YpiRoR6RG/view?usp=drive_link"
    gdown.download(url, ruta_modelo, quiet=False)

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
@st.cache_resource
def cargar_modelo():
    """Cargar modelo YOLO con caché para evitar recargas"""
    try:
        model = YOLO(ruta_modelo)
        st.session_state.modelo_cargado = True
        return model
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None

def procesar_imagen_con_modelo(imagen_pil, confianza_min=0.5):
    """Procesar imagen con modelo YOLO usando PIL en lugar de OpenCV"""
    modelo = cargar_modelo()
    if modelo is None:
        return None, []
    
    try:
        # Convertir PIL a numpy array para YOLO
        img_array = np.array(imagen_pil)
        
        resultados = modelo.predict(
            source=img_array, 
            conf=confianza_min, 
            imgsz=640,
            verbose=False
        )

        # Procesar resultados
        detecciones = []
        img_resultado = None
        
        for r in resultados:
            # Obtener imagen con detecciones dibujadas
            img_resultado_array = r.plot()
            img_resultado = Image.fromarray(img_resultado_array)
            
            # Extraer información de detecciones
            for box in r.boxes:
                clase_id = int(box.cls[0].item())
                clase = modelo.names[clase_id]
                confianza = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                detecciones.append({
                    "clase": clase,
                    "confianza": round(confianza, 3),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })

        return img_resultado, sorted(detecciones, key=lambda x: x['confianza'], reverse=True)
    
    except Exception as e:
        st.error(f"Error procesando imagen: {e}")
        return None, []

def dibujar_detecciones_pil(imagen_pil, detecciones):
    """Dibujar bounding boxes usando PIL en lugar de OpenCV"""
    img_con_boxes = imagen_pil.copy()
    draw = ImageDraw.Draw(img_con_boxes)
    
    try:
        # Intentar cargar una fuente, si falla usar la predeterminada
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for det in detecciones:
        bbox = det['bbox']
        
        # Dibujar rectángulo
        draw.rectangle(
            [(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2'])],
            outline="green",
            width=3
        )
        
        # Etiqueta
        label = f"{det['clase']}: {det['confianza']:.2f}"
        
        # Fondo para el texto
        bbox_text = draw.textbbox((bbox['x1'], bbox['y1'] - 25), label, font=font)
        draw.rectangle(bbox_text, fill="green")
        
        # Texto
        draw.text(
            (bbox['x1'], bbox['y1'] - 25),
            label,
            fill="white",
            font=font
        )
    
    return img_con_boxes

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

def capturar_desde_camara():
    """Función alternativa para captura de cámara usando streamlit_webrtc o similar"""
    st.warning("⚠️ La captura directa desde cámara requiere configuración adicional en Streamlit Cloud")
    st.info("""
    **Opciones para usar la cámara:**
    1. **Deployment local**: Instala opencv-python localmente
    2. **Streamlit Cloud**: Usa la función `st.camera_input()` nativa
    3. **Alternativa**: Sube imágenes tomadas con tu dispositivo
    """)

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
            ["📷 Cámara Streamlit", "📁 Subir imagen", "🎯 Imagen de ejemplo"]
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
        
        if metodo_entrada == "📷 Cámara Streamlit":
            st.subheader("📸 Captura con cámara integrada")
            
            # Usar la función nativa de Streamlit para cámara
            imagen_camara = st.camera_input("Toma una foto")
            
            if imagen_camara is not None:
                imagen_pil = Image.open(imagen_camara)
                st.session_state.imagen_actual = imagen_pil
                st.success("✅ Imagen capturada correctamente")
            
        elif metodo_entrada == "📁 Subir imagen":
            archivo_subido = st.file_uploader(
                "Selecciona una imagen",
                type=['jpg', 'jpeg', 'png'],
                help="Formatos soportados: JPG, JPEG, PNG"
            )
            
            if archivo_subido is not None:
                imagen_pil = Image.open(archivo_subido)
                st.session_state.imagen_actual = imagen_pil
        
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
                img_ejemplo = Image.new('RGB', (640, 480), color=(73, 109, 137))
                st.session_state.imagen_actual = img_ejemplo
                st.info(f"📸 Cargado: {ejemplo_seleccionado} {ejemplos[ejemplo_seleccionado]}")
        
        # Mostrar imagen actual
        if st.session_state.imagen_actual is not None:
            st.subheader("🖼️ Imagen a procesar")
            st.image(st.session_state.imagen_actual, use_column_width=True)
            
            # Botón de procesamiento
            if st.button("🔍 Procesar con CNN", type="primary", use_container_width=True):
                with st.spinner("🧠 Procesando con red neuronal..."):
                    
                    # Ejecutar modelo
                    img_resultado, detecciones = procesar_imagen_con_modelo(
                        st.session_state.imagen_actual, 
                        confianza_min
                    )
                    
                    if img_resultado is not None:
                        # Guardar en historial
                        st.session_state.detecciones_historial.extend(detecciones)
                        
                        # Mostrar resultado
                        if detecciones:
                            st.subheader("🎯 Resultado con detecciones")
                            st.image(img_resultado, use_column_width=True)
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
                        else:
                            st.warning("⚠️ No se detectaron frutas con el umbral de confianza actual")
                    else:
                        st.error("❌ Error al procesar la imagen")
                            
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
        st.markdown("- PIL + YOLO")
        st.markdown("- Ultralytics")
        st.markdown("- Streamlit Dashboard")
        st.markdown("- Procesamiento en tiempo real")
    
    with col2:
        st.markdown("**🎯 Características:**")
        st.markdown("- Modelo YOLO personalizado")
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