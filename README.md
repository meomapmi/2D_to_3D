Đề tài: Nghiên cứu và phát triển ứng dụng biến đổi ảnh 2D thành 3D dựa trên mạng nơ-ron nhân tạo và các kỹ thuật xử lý ảnh hỗ trợ thiết kế đồ họa

Tính năng chính:
- Upload ảnh hoặc video đầu vào (ZIP hoặc MP4).
- Tự động chạy COLMAP để tạo cấu trúc 3D.
- Hiển thị hình ảnh kèm xoay góc nhìn camera.
- Tích điểm 2D trực tiếp trên ảnh.
- Xuất dữ liệu điểm dưới định dạng `.ply`.
- Tích hợp sẵn NeRF/Instant-NGP để tái tạo 3D nâng cao.

Cấu trúc thư mục:
- Source
-- uploads
-- gui.py
-- human.png
-- README.md

Cài đặt
- Local: --gui.py
         --Chạy pip install + tên thư viện --> Thêm các thư viện cần thiết
         -- Cấu hình đường dẫn COLMAP: COLMAP = "C:\\Colmap\\COLMAP.bat"

- Colab:
-- Cài đặt các thư viện cần thiết
!sudo apt-get update
!sudo apt-get install -y build-essential cmake git libgl1-mesa-dev libx11-dev libxi-dev libxrandr-dev libxinerama-dev libxcursor-dev libvulkan-dev

-- Clone Instant NGP
%cd /content
!git clone --recursive https://github.com/NVlabs/instant-ngp.git
%cd instant-ngp

-- Biên dịch
!cmake . -B build -DNGP_BUILD_WITH_GUI=OFF
!cmake --build build --config RelWithDebInfo -j 8

Các công nghệ sử dụng:
- Python
- COLMAP
- Instant NGP
- Open3D, NumPy, Tkinter và các thư viện hỗ trợ khác

Tác giả:
Phạm Khánh Hà Mi
mipham151008@gmail.com

Link Colab: https://colab.research.google.com/drive/1Eyvl1M7Ij7c7y5VRewUgREZuFh8A9y9b?usp=sharing