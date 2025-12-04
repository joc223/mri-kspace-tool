import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. 網頁基本設定
st.set_page_config(page_title="MRI K-space Simulator")

# 2. 【核彈級隱藏 CSS】
# 因為我們把控制項移到中間了，所以可以放心地把整條 Header 藏起來！
hide_all_style = """
<style>
    /* 隱藏頂部 Header (包含 Manage app 按鈕、漢堡選單、側邊欄箭頭) */
    header {visibility: hidden;}
    
    /* 隱藏右上角的三點選單 */
    #MainMenu {visibility: hidden;}
    
    /* 隱藏頁尾 */
    footer {visibility: hidden;}
    
    /* 雙重保險：隱藏 Manage App 按鈕 */
    .stAppDeployButton {display: none;}
    [data-testid="stManageAppButton"] {display: none;}
</style>
"""
st.markdown(hide_all_style, unsafe_allow_html=True)

# 3. 標題與說明
st.title(" MRI K-space 原理模擬器")
st.markdown("""
**K-space (空間頻率)** 與 **影像空間 (Image Space)** 的對應關係觀察：
* **中心點 (0,0)**：代表直流分量 (DC)，訊號最強。
* **$k_x, k_y$**：代表在 X 或 Y 方向上的頻率變化（週期數）。
""")

st.write("---") # 分隔線

# --- 4. 參數控制區 (移到主畫面最上方) ---
# 使用 st.columns 將控制項橫向排列
c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    st.subheader("1. 設定矩陣大小")
    matrix_size = st.selectbox(
        "矩陣大小 (Matrix Size)",
        options=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        index=3
    )

with c2:
    st.subheader("2. 調整 X 頻率")
    kx = st.slider("kx (X 方向週期數)", min_value=-10, max_value=10, value=1, step=1)

with c3:
    st.subheader("3. 調整 Y 頻率")
    ky = st.slider("ky (Y 方向週期數)", min_value=-10, max_value=10, value=0, step=1)

st.write("---") # 分隔線

# --- 5. 核心運算 ---
def generate_centered_pattern(size, k_x, k_y):
    x = np.linspace(-0.5, 0.5, size)
    y = np.linspace(-0.5, 0.5, size)
    X, Y = np.meshgrid(x, y)
    pattern = np.cos(2 * np.pi * (k_x * X + k_y * Y))
    return pattern, x, y

spatial_pattern, x_axis, y_axis = generate_centered_pattern(matrix_size, kx, ky)

# --- 6. 繪圖區域 ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("影像變化（亮暗條紋變化）")
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    
    im = ax1.imshow(spatial_pattern, cmap='gray', 
                    extent=[-0.5, 0.5, -0.5, 0.5], 
                    vmin=-1, vmax=1, origin='lower')
    ax1.scatter([0], [0], color='red', marker='+', s=100, linewidth=2, label='coordinate center')
    
    # 英文標籤 (防亂碼)
    ax1.set_title(f"K-space Point: (kx={kx}, ky={ky})", fontsize=14)
    ax1.set_xlabel("X", fontsize=12)
    ax1.set_ylabel("Y", fontsize=12)
    ax1.legend(loc='upper right')
    
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Signal Intensity', rotation=270, labelpad=15)
    st.pyplot(fig1)

with col_right:
    st.subheader("1D 波形剖面")
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    
    k_magnitude = np.sqrt(kx**2 + ky**2)
    t = np.linspace(-0.5, 0.5, 600)
    
    if k_magnitude == 0:
        waveform = np.ones_like(t)
        info_text = "DC Component (Constant)"
    else:
        waveform = np.cos(2 * np.pi * k_magnitude * t)
        info_text = f"Freq: {k_magnitude:.2f} cycles"

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

    # 底部說明
    st.info(f"""
    **觀察重點：**
    * 目前 K-space 座標： **$k_x={kx}, k_y={ky}$**
    * **左圖**：顯示在影像視野 (FOV) 中，X 方向有 **{abs(kx)}** 個週期，Y 方向有 **{abs(ky)}** 個週期。
    * **右圖**：顯示該頻率的實際正弦波形。中心點 (紅色虛線) 永遠是波峰 (強度=1)，代表相位一致。
    """)