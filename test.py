import numpy as np
import matplotlib.pyplot as plt

def visualize_kspace_point_centered(matrix_size, kx, ky):
    """
    視覺化特定 K-space 點對應的空間頻率圖案 (修正為以中心為原點)。
    """
    
    # --- 修正點：座標從 -0.5 到 0.5，這樣 (0,0) 就會在圖片正中心 ---
    x = np.linspace(-0.5, 0.5, matrix_size)
    y = np.linspace(-0.5, 0.5, matrix_size) # 注意這裡Y軸方向可能因繪圖庫習慣而異，通常需注意上下翻轉
    
    # 建立網格
    X, Y = np.meshgrid(x, y)

    # 計算波形圖案 (這次中心點 x=0, y=0 時，cos(0)=1，會是白色)
    spatial_pattern = np.cos(2 * np.pi * (kx * X + ky * Y))

    # 繪圖設定
    fig = plt.figure(figsize=(14, 6))

    # --- 左圖：2D 空間頻率影像 ---
    ax1 = fig.add_subplot(1, 2, 1)
    # origin='lower' 確保 y軸方向符合直覺 (-0.5 在下, 0.5 在上)
    im = ax1.imshow(spatial_pattern, cmap='gray', extent=[-0.5, 0.5, -0.5, 0.5], vmin=-1, vmax=1, origin='lower')
    ax1.set_title(f'Image Pattern (Centered Origin)\n$k_x$={kx}, $k_y$={ky}', fontsize=14)
    ax1.set_xlabel('X position (FOV)', fontsize=12)
    ax1.set_ylabel('Y position (FOV)', fontsize=12)
    
    # 標示出中心點
    ax1.scatter([0], [0], color='red', marker='+', s=100, label='Isocenter')
    ax1.legend(loc='upper right')
    plt.colorbar(im, ax=ax1, label='Signal Amplitude')

    # --- 右圖：1D 剖面圖 ---
    ax2 = fig.add_subplot(1, 2, 2)
    
    # 沿著波傳遞方向畫剖面
    k_magnitude = np.sqrt(kx**2 + ky**2)
    t = np.linspace(-0.5, 0.5, 1000) # 這裡也改為從中心對稱
    
    if k_magnitude == 0:
        waveform = np.ones_like(t)
    else:
        waveform = np.cos(2 * np.pi * k_magnitude * t)

    ax2.plot(t, waveform, color='blue', linewidth=2)
    ax2.set_title('1D Waveform Profile (Centered)', fontsize=14)
    ax2.set_xlabel('Position (0 is Center)', fontsize=12)
    ax2.set_ylabel('Amplitude', fontsize=12)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 畫一條中心線輔助觀察
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Center (x=0)')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# ==========================================
# 測試修正後的結果
# ==========================================

# 範例：現在 kx=1, ky=0，中心點應該會是「白色」(波峰)
visualize_kspace_point_centered(matrix_size=128, kx=1, ky=0)

# # 範例：kx=2, ky=4
# visualize_kspace_point_centered(matrix_size=128, kx=2, ky=4)