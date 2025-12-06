import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. 網頁基本設定 (移除 layout="wide" 以適配手機)
st.set_page_config(page_title="MRI K-space Simulator")

# 2. 【核彈級隱藏 CSS】
hide_all_style = """
<style>
    /* 隱藏頂部 Header */
    header {visibility: hidden;}
    
    /* 隱藏右上角的三點選單 */
    #MainMenu {visibility: hidden;}
    
    /* 隱藏頁尾 */
    footer {visibility: hidden;}
    
    /* 隱藏 Manage App 按鈕 */
    .stAppDeployButton {display: none;}
    [data-testid="stManageAppButton"] {display: none;}
</style>
"""
st.markdown(hide_all_style, unsafe_allow_html=True)

# 3. 標題與說明
st.title("MRI K-space 原理模擬器")
st.markdown("""
**K-space (空間頻率)** 與 **影像空間 (Image Space)** 的對應關係觀察：
* **中心點 (coordinate center)**：為 kx=0, ky=0 時，訊號最強。
* **$k_x, k_y$**：代表在 X 或 Y 方向上的頻率變化（週期數）。
""")

st.write("---")

# --- 4. 參數控制區 ---
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

st.write("---")

# --- 5. K-space 點陣圖 ---
st.subheader(f"K-space 當前位置示意圖 (Matrix: {matrix_size}x{matrix_size})")

def plot_kspace_grid(k_x, k_y, size):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 限制顯示範圍 (Zoom in)
    display_limit = 10
    
    grid_x, grid_y = np.meshgrid(np.arange(-display_limit, display_limit+1), 
                                 np.arange(-display_limit, display_limit+1))
    
    ax.scatter(grid_x, grid_y, c='yellow', s=80, edgecolors='gray', alpha=0.5, label='Grid')
    
    ax.axhline(0, color='white', linewidth=1)
    ax.axvline(0, color='white', linewidth=1)
    
    if abs(k_x) <= display_limit and abs(k_y) <= display_limit:
        ax.scatter([k_x], [k_y], c='red', s=120, edgecolors='white', linewidth=2, label='Current', zorder=10)
        ax.annotate(f'({k_x}, {k_y})', xy=(k_x, k_y), xytext=(k_x+1, k_y+1),
                    color='white', fontsize=10,
                    arrowprops=dict(facecolor='white', shrink=0.05))
    
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    ax.set_xlabel('kx (Frequency)', color='white', fontsize=10)
    ax.set_ylabel('ky (Phase)', color='white', fontsize=10)
    ax.tick_params(axis='both', colors='white')
    
    ax.set_xlim(-display_limit - 1, display_limit + 1)
    ax.set_ylim(-display_limit - 1, display_limit + 1)
    
    for spine in ax.spines.values():
        spine.set_color('white')
        
    ax.set_title("K-space Sampling Grid", color='white', fontsize=12)
    return fig

st.pyplot(plot_kspace_grid(kx, ky, matrix_size))

# 黃色點點備註
st.warning("""
**💡 備註：**
如果在手機或電腦螢幕上，真的把 128x128 (甚至 4096) 個黃色點點全部畫出來，它們會擠在一起變成一塊「實心的黃色方塊」，會完全看不出「網格」的感覺，因此僅畫到 21x21 的中心區域示意，**絕對完全並非作者本人偷懶**。
""")

st.write("---")

# --- 6. 核心運算 ---
def generate_centered_pattern(size, k_x, k_y):
    x = np.linspace(-0.5, 0.5, size)
    y = np.linspace(-0.5, 0.5, size)
    X, Y = np.meshgrid(x, y)
    pattern = np.cos(2 * np.pi * (k_x * X + k_y * Y))
    return pattern, x, y

spatial_pattern, x_axis, y_axis = generate_centered_pattern(matrix_size, kx, ky)

# --- 7. 下方圖表區 ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("影像變化")
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    
    im = ax1.imshow(spatial_pattern, cmap='gray', 
                    extent=[-0.5, 0.5, -0.5, 0.5], 
                    vmin=-1, vmax=1, origin='lower')
    ax1.scatter([0], [0], color='red', marker='+', s=100, linewidth=2, label='Isocenter')
    
    ax1.set_title(f"Image Space: (kx={kx}, ky={ky})", fontsize=12)
    ax1.set_xlabel("X Position", fontsize=10)
    ax1.set_ylabel("Y Position", fontsize=10)
    ax1.legend(loc='upper right', fontsize='small')
    
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Signal Intensity', rotation=270, labelpad=15)
    st.pyplot(fig1)
    
    # 【備註 1】解釋 kx, ky
    st.info(f"""
    **現在是 $k_x={kx}, k_y={ky}$**
    這代表在 X 方向有 **{abs(kx)}** 個週期的亮暗條紋變化，
    而 Y 方向有 **{abs(ky)}** 個週期的亮暗條紋變化。
    """)

with col_right:
    st.subheader("1D 波形剖面")
    fig2, ax2 = plt.subplots(figsize=(6, 4)) # 高度稍微調高，視覺平衡
    
    k_magnitude = np.sqrt(kx**2 + ky**2)
    t = np.linspace(-0.5, 0.5, 600)
    
    if k_magnitude == 0:
        waveform = np.ones_like(t)
        info_text = "DC Component (Constant)"
    else:
        waveform = np.cos(2 * np.pi * k_magnitude * t)
        info_text = f"Freq: {k_magnitude:.2f}"

    ax2.plot(t, waveform, color='#1f77b4', linewidth=2)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.6, label='Center')
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlim(-0.5, 0.5)
    
    ax2.set_xlabel("Position", fontsize=10)
    ax2.set_ylabel("Amplitude", fontsize=10)
    ax2.set_title(f"Profile: {info_text}", fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize='small')
    st.pyplot(fig2)
    
    # 【備註 2】解釋波形為何不斜
    st.info("""
    **💡 為什麼波形不是斜的？**
    這張圖顯示的是 **訊號強度 (Amplitude)** 的變化，而非空間幾何形狀。
    無論左圖的條紋是直的、橫的或斜的，沿著波傳遞方向切開來看，其亮暗強度的變化（由白變黑再變白）永遠呈現上下震盪的正弦波形。
    """)

# 底部總結
st.success("""
**總結觀察重點：**
1. **上方黑底圖**：顯示您目前在 K-space 的取樣位置（紅色點）。
2. **左圖 (影像)**：顯示該頻率對應的空間條紋方向與密度。
3. **右圖 (波形)**：顯示該頻率的實際震盪情形。中心點 (紅色虛線) 永遠是波峰，代表相位一致。
""")