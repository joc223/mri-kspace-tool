import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. 網頁基本設定
st.set_page_config(page_title="MRI K-space Simulator", layout="wide")

# 2. 【最終防禦版 CSS】
hide_css = """
<style>
    /* 1. 隱藏右上角的三點選單 (別人無法按 View Source) */
    #MainMenu {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    
    /* 2. 隱藏頁尾的 Made with Streamlit */
    footer {visibility: hidden;}
    
    /* 3. 隱藏右下角的 Manage App 按鈕 (這對訪客有效) */
    .stAppDeployButton {display: none;}
    [data-testid="stManageAppButton"] {display: none;}
    
    /* 4. 針對 Embed 模式底下的 Bar 也強制隱藏 (以防萬一) */
    .viewerBadge_container__1QSob {display: none;}
</style>
"""
st.markdown(hide_css, unsafe_allow_html=True)

# 3. 標題與說明
st.title(" MRI K-space 原理互動模擬器")
st.markdown("""
**操作說明：**
* 請展開左側選單調整 **$k_x$** 與 **$k_y$**。
* **左圖**：顯示 K-space 該點對應的 **空間條紋 (Spatial Pattern)**。
* **右圖**：顯示沿著波傳遞方向的 **1D 波形 (Waveform)**。
""")

# --- 側邊欄 ---
st.sidebar.header("1. 參數設定 (Parameters)")

matrix_size = st.sidebar.selectbox(
    "選擇矩陣大小 (Matrix Size)",
    options=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    index=3
)

st.sidebar.write("---")

st.sidebar.subheader("2. 調整 K-space 座標")
kx = st.sidebar.slider("kx (X 方向週期數)", min_value=-10, max_value=10, value=1, step=1)
ky = st.sidebar.slider("ky (Y 方向週期數)", min_value=-10, max_value=10, value=0, step=1)

# --- 核心運算 ---
def generate_centered_pattern(size, k_x, k_y):
    x = np.linspace(-0.5, 0.5, size)
    y = np.linspace(-0.5, 0.5, size)
    X, Y = np.meshgrid(x, y)
    pattern = np.cos(2 * np.pi * (k_x * X + k_y * Y))
    return pattern, x, y

spatial_pattern, x_axis, y_axis = generate_centered_pattern(matrix_size, kx, ky)

# --- 繪圖 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(" 空間域影像 (Image Pattern)")
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    im = ax1.imshow(spatial_pattern, cmap='gray', 
                    extent=[-0.5, 0.5, -0.5, 0.5], 
                    vmin=-1, vmax=1, origin='lower')
    ax1.scatter([0], [0], color='red', marker='+', s=100, linewidth=2, label='Isocenter')
    
    # 英文標籤 (防亂碼)
    ax1.set_title(f"K-space Point: (kx={kx}, ky={ky})", fontsize=14)
    ax1.set_xlabel("X Position (FOV)", fontsize=12)
    ax1.set_ylabel("Y Position (FOV)", fontsize=12)
    ax1.legend(loc='upper right')
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Signal Intensity', rotation=270, labelpad=15)
    st.pyplot(fig1)

with col2:
    st.subheader(" 1D 波形剖面 (Waveform)")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    k_magnitude = np.sqrt(kx**2 + ky**2)
    t = np.linspace(-0.5, 0.5, 600)
    
    if k_magnitude == 0:
        waveform = np.ones_like(t)
        info_text = "DC Component (Constant)"
    else:
        waveform = np.cos(2 * np.pi * k_magnitude * t)
        info_text = f"Freq: {k_magnitude:.2f} cycles/FOV"

    ax2.plot(t, waveform, color='#1f77b4', linewidth=2)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.6, label='Center (x=0)')
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_xlabel("Position along wave direction", fontsize=12)
    ax2.set_ylabel("Amplitude", fontsize=12)
    ax2.set_title(f"Waveform Profile ({info_text})", fontsize=14)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    st.pyplot(fig2)

    st.info(f"""
    **觀察重點：**
    * 目前 K-space 點選在 **$k_x={kx}, k_y={ky}$**。
    * 這代表在影像視野中，存在 **X方向 {abs(kx)} 個週期** 與 **Y方向 {abs(ky)} 個週期** 的變化。
    * 請看紅色中心線，該處訊號強度為 **{waveform[len(t)//2]:.1f}** (1.0 代表全白)，這驗證了中心點相位一致的特性。
    """)