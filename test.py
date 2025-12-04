import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. 網頁基本設定
st.set_page_config(page_title="MRI K-space Simulator", layout="wide")

# 2. 【隱藏程式碼與選單】的 CSS 語法
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stAppDeployButton {display:none;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# 3. 標題與說明 (網頁介面保留中文，同學比較親切)
st.title(" MRI K-space 原理互動模擬器")
st.markdown("""
透過此工具觀察 **K-space (空間頻率)** 上的點如何對應到 **影像空間 (Image Space)** 的條紋圖案。
* **$k_x, k_y$**：代表在 X 或 Y 方向上，一個 FOV 內變化的週期數。
* **中心點**：座標 (0,0) 代表直流分量 (DC)，訊號最強且恆定。
""")

# --- 側邊欄：參數設定 ---
st.sidebar.header("1. 參數設定 (Parameters)")

# 矩陣大小 (您修改過的部分：增加了更多選項)
matrix_size = st.sidebar.selectbox(
    "選擇矩陣大小 (Matrix Size)",
    options=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    index=3  # 預設 128
)

st.sidebar.write("---")

# kx, ky 滑桿
st.sidebar.subheader("2. 調整 K-space 座標")
kx = st.sidebar.slider("kx (X 方向週期數)", min_value=-10, max_value=10, value=1, step=1)
ky = st.sidebar.slider("ky (Y 方向週期數)", min_value=-10, max_value=10, value=0, step=1)

# --- 核心運算 (修正為中心原點) ---
def generate_centered_pattern(size, k_x, k_y):
    # 建立從 -0.5 到 0.5 的網格
    # 這樣 (0,0) 就會在矩陣的正中心
    x = np.linspace(-0.5, 0.5, size)
    y = np.linspace(-0.5, 0.5, size)
    X, Y = np.meshgrid(x, y)
    
    # 計算波形: cos(2 * pi * (kx*x + ky*y))
    # 當 x=0, y=0 時，cos(0) = 1 (白色)，符合講義描述
    pattern = np.cos(2 * np.pi * (k_x * X + k_y * Y))
    return pattern, x, y

spatial_pattern, x_axis, y_axis = generate_centered_pattern(matrix_size, kx, ky)

# --- 繪圖區域 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(" 空間域影像 (Image Pattern)")
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    
    # 顯示影像 (origin='lower' 讓 Y軸由下往上增加)
    im = ax1.imshow(spatial_pattern, cmap='gray', 
                    extent=[-0.5, 0.5, -0.5, 0.5], 
                    vmin=-1, vmax=1, origin='lower')
    
    # 標示中心點
    ax1.scatter([0], [0], color='red', marker='+', s=100, linewidth=2, label='Isocenter')
    
    # 【修改重點】圖表內的文字改為英文，避免亂碼
    ax1.set_title(f"K-space: (kx={kx}, ky={ky})", fontsize=14)
    ax1.set_xlabel("X Position (FOV)", fontsize=12)
    ax1.set_ylabel("Y Position (FOV)", fontsize=12)
    ax1.legend(loc='upper right')
    
    # Colorbar 標籤也改英文
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Signal Intensity', rotation=270, labelpad=15)
    
    st.pyplot(fig1)

with col2:
    st.subheader(" 1D 波形剖面 (Waveform)")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    
    # 計算合成頻率
    k_magnitude = np.sqrt(kx**2 + ky**2)
    
    # 產生高解析度座標畫波形
    t = np.linspace(-0.5, 0.5, 600)
    
    if k_magnitude == 0:
        waveform = np.ones_like(t)
        info_text = "DC Component (Constant)" # 改英文
    else:
        waveform = np.cos(2 * np.pi * k_magnitude * t)
        info_text = f"Freq: {k_magnitude:.2f} cycles/FOV" # 改英文

    ax2.plot(t, waveform, color='#1f77b4', linewidth=2)
    
    # 標示中心線
    ax2.axvline(0, color='red', linestyle='--', alpha=0.6, label='Center (x=0)')
    
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlim(-0.5, 0.5)
    
    # 【修改重點】圖表內的文字改為英文
    ax2.set_xlabel("Position along wave direction", fontsize=12)
    ax2.set_ylabel("Amplitude", fontsize=12)
    ax2.set_title(f"Waveform ({info_text})", fontsize=14)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    st.pyplot(fig2)

    # 文字說明區 (這部分在網頁上顯示，可以用中文)
    st.info(f"""
    **觀察重點：**
    * 目前 K-space 點選在 **$k_x={kx}, k_y={ky}$**。
    * 這代表在影像視野中，存在 **X方向 {abs(kx)} 個週期** 與 **Y方向 {abs(ky)} 個週期** 的變化。
    * 請看紅色中心線，該處訊號強度為 **{waveform[len(t)//2]:.1f}** (1.0 代表全白)，這驗證了中心點相位一致的特性。
    """)
