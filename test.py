import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. 網頁基本設定
# 設定網頁標題與圖示
st.set_page_config(page_title="MRI K-space Simulator")

# 2. CSS 樣式設定
# 隱藏 Streamlit 預設的選單、頁尾等元素，讓介面更乾淨
hide_all_style = """
<style>
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {display: none;}
    [data-testid="stManageAppButton"] {display: none;}
    
    /* 優化 Tabs 字體大小與粗細 */
    button[data-baseweb="tab"] {
        font-size: 18px;
        font-weight: bold;
    }
</style>
"""
st.markdown(hide_all_style, unsafe_allow_html=True)

# 3. 主標題
st.title("MRI K-space 原理模擬器")

# 建立兩個分頁：模擬器與原理教學
tab_sim, tab_theory = st.tabs(["K-space 模擬器", "空間編碼原理"]) 


# ==========================================
# 分頁 1 : K-space、亮暗條紋變化、波形
# ==========================================
with tab_sim:
    # 顯示說明文字
    st.markdown("""
    **本分頁用於觀察 K-space、條紋變化與波形 的對應關係：**
    * **K-space 中心點 (coordinate center)**：為 $k_x, k_y$ 時，訊號最強。
    * **$k_x, k_y$**：代表在 K-space 上 X 或 Y 方向上的頻率變化 (或稱亮暗條紋變化）。
    """)
    st.write("---")

    # --- 參數控制區 (分三欄) ---
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        st.subheader("1. 設定矩陣大小")
        # 下拉選單選擇矩陣大小
        matrix_size = st.selectbox(
            "矩陣大小 (Matrix Size)",
            options=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
            index=3
        )

    with c2:
        st.subheader("2. 調整 X 方向的頻率")
        # 滑桿控制 kx 數值
        kx = st.slider("kx (可當作頻率編碼)", min_value=-10, max_value=10, value=1, step=1)

    with c3:
        st.subheader("3. 調整 Y 方向的頻率")
        # 滑桿控制 ky 數值
        ky = st.slider("ky (可當作相位編碼)", min_value=-10, max_value=10, value=0, step=1)

    st.write("---")

    # --- K-space 點陣圖繪製 ---
    st.subheader(f"K-space 目前的位置 (Matrix size : {matrix_size}x{matrix_size})")

    # 定義繪圖函式
    def plot_kspace_grid(k_x, k_y, size):
        fig, ax = plt.subplots(figsize=(6, 4))
        display_limit = 10 # 只顯示中心區域 +/- 10
        
        # 產生網格座標
        grid_x, grid_y = np.meshgrid(np.arange(-display_limit, display_limit+1), 
                                     np.arange(-display_limit, display_limit+1))
        
        # 畫出背景的黃色網格點
        ax.scatter(grid_x, grid_y, c='yellow', s=80, edgecolors='gray', alpha=0.5, label='Grid')
        
        # 畫出中心十字線
        ax.axhline(0, color='white', linewidth=1)
        ax.axvline(0, color='white', linewidth=1)
        
        # 畫出目前選定的紅色點
        if abs(k_x) <= display_limit and abs(k_y) <= display_limit:
            ax.scatter([k_x], [k_y], c='red', s=120, edgecolors='white', linewidth=2, label='Current', zorder=10)
            ax.annotate(f'({k_x}, {k_y})', xy=(k_x, k_y), xytext=(k_x+1, k_y+1),
                        color='white', fontsize=10,
                        arrowprops=dict(facecolor='white', shrink=0.05))
        
        # 設定圖表樣式 (黑底)
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.set_xlabel('kx (Frequency)', color='white', fontsize=10)
        ax.set_ylabel('ky (Phase)', color='white', fontsize=10)
        ax.tick_params(axis='both', colors='white')
        ax.set_xlim(-display_limit - 1, display_limit + 1)
        ax.set_ylim(-display_limit - 1, display_limit + 1)
        
        for spine in ax.spines.values():
            spine.set_color('white')
            
        ax.set_title("K-space", color='white', fontsize=12)
        return fig

    # 顯示 K-space 圖
    st.pyplot(plot_kspace_grid(kx, ky, matrix_size))

    # 顯示備註警示
    st.warning("""
    **備註：**
    如果在手機或電腦螢幕上，真的把 128x128 (甚至是 4096) 個黃色點全部畫出來，
                它們會擠在一起變成一塊「實心的黃色方塊」，會完全看不出「網格」的感覺，
                因此僅畫到 21x21 作為示意，**絕對並非作者本人想偷懶**。
    """)

    st.write("---")

    # --- 核心運算：產生空間圖案 ---
    # 使用快取裝飾器加速運算
    @st.cache_data 
    def generate_centered_pattern(size, k_x, k_y):
        x = np.linspace(-0.5, 0.5, size)
        y = np.linspace(-0.5, 0.5, size)
        X, Y = np.meshgrid(x, y)
        # 計算 2D Cosine 波形
        pattern = np.cos(2 * np.pi * (k_x * X + k_y * Y))
        return pattern

    # 執行運算
    spatial_pattern = generate_centered_pattern(matrix_size, kx, ky)

    # --- 下方圖表區 (左右兩欄) ---
    col_left, col_right = st.columns([1, 1])

    # 左欄：影像變化 (條紋圖)
    with col_left:
        st.subheader("影像變化 (亮暗條紋變化)")
        fig1, ax1 = plt.subplots(figsize=(7, 7))
        
        # 顯示影像
        im = ax1.imshow(spatial_pattern, cmap='gray', 
                        extent=[-0.5, 0.5, -0.5, 0.5], 
                        vmin=-1, vmax=1, origin='lower')
        # 標示中心點
        ax1.scatter([0], [0], color='red', marker='+', s=100, linewidth=2, label='coordinate center')
        
        # 設定標題與座標軸
        ax1.set_title(f"Image Space : (kx={kx}, ky={ky})", fontsize=12)
        ax1.set_xlabel("X Position", fontsize=10)
        ax1.set_ylabel("Y Position", fontsize=10)
        ax1.legend(loc='upper right', fontsize='small')
        
        # 加入 Colorbar
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Signal Intensity', rotation=270, labelpad=15)
        st.pyplot(fig1)
        
        # 顯示參數說明
        st.info(f"""
        **說明 : 現在是 $k_x={kx}, k_y={ky}$**
        ，這代表在 X 方向有 **{abs(kx)}** 個週期的亮暗條紋變化，
        而 Y 方向有 **{abs(ky)}** 個週期的亮暗條紋變化。
        """)

    # 右欄：1D 波形剖面
    with col_right:
        st.subheader("1D 波形剖面")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        
        # 計算合成頻率大小
        k_magnitude = np.sqrt(kx**2 + ky**2)
        t = np.linspace(-0.5, 0.5, 600)
        
        # 根據頻率產生波形
        if k_magnitude == 0:
            waveform = np.ones_like(t) # 直流分量
            info_text = "k-space 中心點訊號最強，擁有最大亮度"
        else:
            waveform = np.cos(2 * np.pi * k_magnitude * t)
            info_text = f"Freq = {k_magnitude:.2f} cycles per unit distance"

        # 繪製波形
        ax2.plot(t, waveform, color='#1f77b4', linewidth=2)
        ax2.axvline(0, color='red', linestyle='--', alpha=0.6, label='Center')
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_xlim(-0.5, 0.5)
        
        # 設定標題與格線
        ax2.set_xlabel("Position", fontsize=10)
        ax2.set_ylabel("Amplitude", fontsize=10)
        ax2.set_title(f"Profile : {info_text}", fontsize=12)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(fontsize='small')
        st.pyplot(fig2)
        
        # 顯示波形觀念說明
        st.info("""
        **說明 : 為什麼有相位編碼時波形不是斜的？**
        這張圖顯示的是 **「訊號強度 (Amplitude)」** 的變化，而非空間的幾何形狀。
        無論條紋是直的、橫的或斜的，沿著波傳遞方向切開來看，
        其強度變化永遠呈現上下震盪的正弦波形。
        """)


# ==========================================
# 分頁 2 : 空間編碼原理
# ==========================================
with tab_theory:
    st.header("空間編碼原理模擬器")
    
    # 原理文字說明
    st.write("""
    **原理說明：**
    底下模擬**空間編碼 (梯度磁場) 如何讓不同位置的質子產生相位差**，最終形成訊號波形。
             
    補充 : 「相位編碼」是透過 **相位差** ；「頻率編碼」是透過 **速度差** ，來達成空間定位的目的。
    1. **梯度 (Gradient)**：在不同位置，施加不同磁場梯度。
    2. **相角 (Phase)**：因為不同的梯度，導致質子產生不同旋進頻率 (就是不同的轉速)。
    3. **信號強度 (Projection)**：將相角投影到 Z 軸（就是相角的 Z 軸向量、投影）。
    4. **波形 (Waveform)**：將投影量連起來，就變成了 Cosine 波形！
    """)
    
    # 梯度強度滑桿
    pe_gradient = st.slider("調整空間編碼梯度強度", -5.0, 5.0, 2.0, step=0.5)
    
    # 設定圖表 (有4層)，高度加大到 16 以容納四張圖
    fig_pe, (ax_grad, ax_spins, ax_proj, ax_wave) = plt.subplots(4, 1, figsize=(8, 16), 
                                                                 gridspec_kw={'height_ratios': [1, 1.2, 1.2, 1]})
    fig_pe.subplots_adjust(hspace=0.6) # 拉開子圖間距
    
    # --- 1. 第一層：空間編碼梯度強度層 ---
    y_pos = np.linspace(-1, 1, 21) # 設定 21 個取樣點
    field_strength = pe_gradient * y_pos # 計算各點磁場強度
    
    # 繪製梯度斜線
    ax_grad.plot(y_pos, field_strength, color='lime', linewidth=1.5, alpha=0.8)
    ax_grad.axhline(0, color='white', linestyle='--', alpha=0.5)
    
    # 畫出梯度箭頭 (綠色)
    for y, f in zip(y_pos[::2], field_strength[::2]):
        ax_grad.arrow(y, 0, 0, f, 
                      head_width=0.06, head_length=0.3, 
                      length_includes_head=True, 
                      fc='lime', ec='lime', width=0.012)

    # 設定第一層樣式
    ax_grad.set_facecolor('black')
    ax_grad.set_title(f"1. Gradient Field Strength (Slope = {pe_gradient})", color='white', fontsize=12, pad=10)
    ax_grad.set_ylabel("G strength", color='white')
    ax_grad.tick_params(colors='white')
    ax_grad.set_ylim(-6, 6)
    
    # --- 2. 第二層：自旋相位角 ---
    ax_spins.set_facecolor('black')
    ax_spins.set_xlim(-1.2, 1.2)
    ax_spins.set_ylim(-0.6, 0.6)
    ax_spins.axis('on')
    ax_spins.set_yticks([]) # 隱藏 Y 軸刻度
    for spine in ax_spins.spines.values(): spine.set_color('white')

    # 計算各點相位角
    phase_angles = -pe_gradient * y_pos * np.pi 
    for i, y in enumerate(y_pos):
        center_x = y; center_y = 0
        # 畫圓圈
        circle = plt.Circle((center_x, center_y), 0.04, color='gray', fill=False)
        ax_spins.add_artist(circle)
        # 計算指針方向
        dx = 0.04 * np.sin(phase_angles[i])
        dy = 0.04 * np.cos(phase_angles[i])
        # 畫出相位指針 (藍色)
        ax_spins.arrow(center_x, center_y, dx, dy, head_width=0.0, color='cyan', width=0.008)
    
    # 設定第二層標題
    ax_spins.set_title("2. Spin Phase Angle", color='white', fontsize=12, pad=10)
    ax_spins.set_xlabel("Position", color='white')
    ax_spins.tick_params(axis='x', colors='white')

    # --- 3. 第三層：信號強度投影量 ---
    ax_proj.set_facecolor('black')
    ax_proj.set_xlim(-1.2, 1.2)
    ax_proj.set_ylim(-0.6, 0.6)
    ax_proj.axis('on')
    ax_proj.set_yticks([]) 
    for spine in ax_proj.spines.values(): spine.set_color('white')

    for i, y in enumerate(y_pos):
        center_x = y; center_y = 0
        # 畫圓圈
        circle = plt.Circle((center_x, center_y), 0.04, color='gray', fill=False)
        ax_proj.add_artist(circle)
        
        # 計算垂直投影分量 (Cosine 值)
        proj_dy = 0.04 * np.cos(phase_angles[i])
        
        # 畫垂直箭頭 (黃色)，代表訊號強度
        ax_proj.arrow(center_x, center_y, 0, proj_dy, head_width=0.0, color='yellow', width=0.005)

    # 設定第三層標題
    ax_proj.set_title("3. Signal Intensity", color='white', fontsize=12, pad=10)
    ax_proj.set_xlabel("Position", color='white')
    ax_proj.tick_params(axis='x', colors='white')

    # --- 4. 第四層：Cosine 波形 ---
    # 使用更多點數 (300點) 讓波形平滑
    y_smooth = np.linspace(-1, 1, 300)
    phase_smooth = -pe_gradient * y_smooth * np.pi
    wave_smooth = np.cos(phase_smooth)
    
    ax_wave.set_facecolor('black')
    # 畫出黃色連續波形
    ax_wave.plot(y_smooth, wave_smooth, color='yellow', linewidth=2)
    # 填色增加視覺效果
    ax_wave.fill_between(y_smooth, wave_smooth, color='yellow', alpha=0.3)
    
    # 設定第四層樣式
    ax_wave.set_title("4. Resulting Waveform (Cosine)", color='white', fontsize=12, pad=10)
    ax_wave.set_ylabel("Intensity", color='white')
    ax_wave.set_xlabel("Position", color='white')
    ax_wave.tick_params(colors='white')
    ax_wave.set_ylim(-1.2, 1.2)

    # 設定整張畫布背景為黑色
    fig_pe.patch.set_facecolor('black')
    st.pyplot(fig_pe)

# 頁尾說明
st.divider()
st.caption("2025 CSMU MIRS 1298002 wcy. 如果有發現未修復的 Bug 或建議，請聯絡 xes67421@gmail.com 謝謝。")

