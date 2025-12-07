import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. 網頁基本設定
st.set_page_config(page_title="MRI K-space Simulator")

# 2. 【核彈級隱藏 CSS】
hide_all_style = """
<style>
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {display: none;}
    [data-testid="stManageAppButton"] {display: none;}
    
    /* 優化 Tabs 字體 */
    button[data-baseweb="tab"] {
        font-size: 18px;
        font-weight: bold;
    }
</style>
"""
st.markdown(hide_all_style, unsafe_allow_html=True)

# 3. 標題
st.title("MRI K-space 原理模擬器")

# --- 建立分頁 (Tabs) ---
tab_sim, tab_theory = st.tabs(["🧲 K-space 模擬器", "📚 相位編碼原理"])

# ==========================================
# 分頁 1: K-space 模擬器
# ==========================================
with tab_sim:
    st.markdown("""
    **觀察 K-space (空間頻率) 與 影像空間 (Image Space) 的對應關係：**
    * **中心點 (coordinate center)**：為 kx=0, ky=0 時，訊號最強。
    * **$k_x, k_y$**：代表在 X 或 Y 方向上的頻率變化（週期數）。
    """)
    st.write("---")

    # --- 參數控制區 ---
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
        kx = st.slider("kx (可當作頻率編碼)", min_value=-10, max_value=10, value=1, step=1)

    with c3:
        st.subheader("3. 調整 Y 頻率")
        ky = st.slider("ky (可當作相位編碼)", min_value=-10, max_value=10, value=0, step=1)

    st.write("---")

    # --- K-space 點陣圖 ---
    st.subheader(f"K-space 當前位置示意圖 (Matrix: {matrix_size}x{matrix_size})")

    def plot_kspace_grid(k_x, k_y, size):
        fig, ax = plt.subplots(figsize=(6, 4))
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
            
        ax.set_title("K-space Sampling Grid (Zoomed-in View)", color='white', fontsize=12)
        return fig

    st.pyplot(plot_kspace_grid(kx, ky, matrix_size))

    st.warning("""
    **備註：**
    如果在手機或電腦螢幕上，真的把 128x128 (甚至 4096) 個黃色點點全部畫出來，它們會擠在一起變成一塊「實心的黃色方塊」，會完全看不出「網格」的感覺，因此僅畫到 21x21 的中心區域示意，**絕對完全並非作者本人偷懶**。
    """)

    st.write("---")

    # --- 核心運算 ---
    @st.cache_data 
    def generate_centered_pattern(size, k_x, k_y):
        x = np.linspace(-0.5, 0.5, size)
        y = np.linspace(-0.5, 0.5, size)
        X, Y = np.meshgrid(x, y)
        pattern = np.cos(2 * np.pi * (k_x * X + k_y * Y))
        return pattern

    spatial_pattern = generate_centered_pattern(matrix_size, kx, ky)

    # --- 下方圖表區 ---
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
        
        st.info(f"""
        **現在是 $k_x={kx}, k_y={ky}$**
        這代表在 X 方向有 **{abs(kx)}** 個週期的亮暗條紋變化，
        而 Y 方向有 **{abs(ky)}** 個週期的亮暗條紋變化。
        """)

    with col_right:
        st.subheader("1D 波形剖面")
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
        ax2.axvline(0, color='red', linestyle='--', alpha=0.6, label='Center')
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_xlim(-0.5, 0.5)
        
        ax2.set_xlabel("Position", fontsize=10)
        ax2.set_ylabel("Amplitude", fontsize=10)
        ax2.set_title(f"Profile: {info_text}", fontsize=12)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(fontsize='small')
        st.pyplot(fig2)
        
        st.info("""
        **為什麼波形不是斜的？**
        這張圖顯示的是 **「訊號強度 (Amplitude)」** 的變化，而非空間幾何形狀。
        無論左圖的條紋是直的、橫的或斜的，沿著波傳遞方向切開來看，
        其亮暗強度的變化（由白變黑再變白）永遠呈現上下震盪的正弦波形。
        """)

# ==========================================
# 分頁 2: 原理教學 (直接顯示，移除 expander)
# ==========================================
with tab_theory:
    st.header("📚 進階原理教學：相位編碼")
    
    st.write("""
    **原理說明：**
    這張圖模擬了 **梯度磁場 ($G_y$)** 如何讓不同位置的質子產生相位差，最終形成訊號波形。
    1. **梯度 (Gradient)**：施加磁場梯度。
    2. **相角 (Phase)**：質子產生不同角度的旋轉。
    3. **投影 (Projection)**：取出垂直方向的分量（實際訊號強度）。
    4. **波形 (Waveform)**：將投影量連起來，就變成了 Cosine 波形！
    """)
    
    pe_gradient = st.slider("調整相位編碼梯度強度 ($G_y$)", -5.0, 5.0, 2.0, step=0.5)
    
    # 設定圖表 (4層)，高度加大到 16 以容納四張圖
    fig_pe, (ax_grad, ax_spins, ax_proj, ax_wave) = plt.subplots(4, 1, figsize=(8, 16), 
                                                                 gridspec_kw={'height_ratios': [1, 1.2, 1.2, 1]})
    fig_pe.subplots_adjust(hspace=0.6) # 拉開間距
    
    # --- 1. 第一層：梯度層 (ax_grad) ---
    y_pos = np.linspace(-1, 1, 21)
    field_strength = pe_gradient * y_pos
    ax_grad.plot(y_pos, field_strength, color='lime', linewidth=1.5, alpha=0.8)
    ax_grad.axhline(0, color='white', linestyle='--', alpha=0.5)
    
    # 畫梯度箭頭
    for y, f in zip(y_pos[::2], field_strength[::2]):
        ax_grad.arrow(y, 0, 0, f, 
                      head_width=0.06, head_length=0.3, 
                      length_includes_head=True, 
                      fc='lime', ec='lime', width=0.012)

    ax_grad.set_facecolor('black')
    ax_grad.set_title(f"1. Gradient Field Strength (Slope = {pe_gradient})", color='white', fontsize=12, pad=10)
    ax_grad.set_ylabel("G strength", color='white')
    ax_grad.tick_params(colors='white')
    ax_grad.set_ylim(-6, 6)
    
    # --- 2. 第二層：相位角 (ax_spins) ---
    ax_spins.set_facecolor('black')
    ax_spins.set_xlim(-1.2, 1.2)
    ax_spins.set_ylim(-0.6, 0.6)
    ax_spins.axis('on')
    ax_spins.set_yticks([]) 
    for spine in ax_spins.spines.values(): spine.set_color('white')

    phase_angles = -pe_gradient * y_pos * np.pi 
    for i, y in enumerate(y_pos):
        center_x = y; center_y = 0
        circle = plt.Circle((center_x, center_y), 0.04, color='gray', fill=False)
        ax_spins.add_artist(circle)
        dx = 0.04 * np.sin(phase_angles[i])
        dy = 0.04 * np.cos(phase_angles[i])
        # 相位指針 (藍色)
        ax_spins.arrow(center_x, center_y, dx, dy, head_width=0.0, color='cyan', width=0.008)
    
    ax_spins.set_title("2. Spin Phase Angle (Rotating Vectors)", color='white', fontsize=12, pad=10)
    ax_spins.set_xlabel("Position Y", color='white')
    ax_spins.tick_params(axis='x', colors='white')

    # --- 3. 第三層：信號投影量 (ax_proj) ---
    ax_proj.set_facecolor('black')
    ax_proj.set_xlim(-1.2, 1.2)
    ax_proj.set_ylim(-0.6, 0.6)
    ax_proj.axis('on')
    ax_proj.set_yticks([]) 
    for spine in ax_proj.spines.values(): spine.set_color('white')

    for i, y in enumerate(y_pos):
        center_x = y; center_y = 0
        circle = plt.Circle((center_x, center_y), 0.04, color='gray', fill=False)
        ax_proj.add_artist(circle)
        
        # 計算垂直投影分量
        proj_dy = 0.04 * np.cos(phase_angles[i])
        
        # 畫垂直箭頭 (黃色) - 稍微加長加細
        ax_proj.arrow(center_x, center_y, 0, proj_dy, head_width=0.0, color='yellow', width=0.005)

    ax_proj.set_title("3. Signal Intensity (Vertical Projection)", color='white', fontsize=12, pad=10)
    ax_proj.set_xlabel("Position Y", color='white')
    ax_proj.tick_params(axis='x', colors='white')

    # --- 4. 第四層：Cosine 波形 (ax_wave) ---
    y_smooth = np.linspace(-1, 1, 300)
    phase_smooth = -pe_gradient * y_smooth * np.pi
    wave_smooth = np.cos(phase_smooth)
    
    ax_wave.set_facecolor('black')
    # 畫出黃色波形
    ax_wave.plot(y_smooth, wave_smooth, color='yellow', linewidth=2)
    # 填色增加視覺效果
    ax_wave.fill_between(y_smooth, wave_smooth, color='yellow', alpha=0.3)
    
    ax_wave.set_title("4. Resulting Waveform (Cosine)", color='white', fontsize=12, pad=10)
    ax_wave.set_ylabel("Intensity", color='white')
    ax_wave.set_xlabel("Position Y", color='white')
    ax_wave.tick_params(colors='white')
    ax_wave.set_ylim(-1.2, 1.2)

    fig_pe.patch.set_facecolor('black')
    st.pyplot(fig_pe)

st.divider()
st.caption("© 2025 CSMU MIRS 1298002 wcy. 如果有發現未修復的 Bug 或建議，請聯絡 xes67421@gmail.com 謝謝。")

