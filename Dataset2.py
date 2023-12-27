import cv2
import numpy as np
import time
import os

# Inisialisasi kamera
cap = cv2.VideoCapture(0)  # Ganti angka 0 dengan indeks kamera jika memiliki lebih dari satu kamera

j = 1

# Tentukan ranks dan suits untuk setiap putaran
ranks = [str(i) for i in range(2, 11)] + ["jack", "queen", "king", "ace"]
suits = ["diamonds", "hearts", "spades", "clubs"]

# Pilih indeks peringkat dan jenis kartu awal
current_rank_index = 0
current_suit_index = 0
# Ukuran kartu remi (lebar x tinggi)
card_width, card_height = 240, 360

# Menghitung koordinat untuk menempatkan kotak di tengah frame
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
roi_x = (frame_width - card_width) // 2
roi_y = (frame_height - card_height) // 2

# Loop untuk mengambil gambar dari kamera
while True:
    # Ambil frame dari kamera
    ret, frame = cap.read()

    # Pilih ROI (Region of Interest) pada tengah frame
    roi = frame[roi_y:roi_y + card_height, roi_x:roi_x + card_width]
    
    # Konversi ROI ke skala abu-abu dan ukuran yang lebih kecil
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, (128, 128), interpolation=cv2.INTER_AREA)

    # Tampilkan ROI yang telah diproses
    cv2.imshow("Processed ROI", roi_resized)

    # Duplikat frame untuk menggambar kotak pada frame asli
    copy = frame.copy()

    # Gambar kotak pada frame
    cv2.rectangle(copy, (roi_x, roi_y), (roi_x + card_width, roi_y + card_height), (0, 0, 0), 5)

    # Tampilkan frame
    cv2.imshow("Capture Images - Press 'P' to capture, 'O' to change rank", copy)

    # Tangkap tombol keyboard
    key = cv2.waitKey(1)

    # Tombol 'p' untuk mengambil gambar
    if key == ord('p'):
        rank = ranks[current_rank_index]
        suit = suits[current_suit_index]

        # Nama file gambar
        file_name = f"{rank}_{suit}_{j}.jpg"

        # Cek apakah file dengan nama tersebut sudah ada
        while os.path.exists(os.path.join("dataset", f"{rank}_{suit}", file_name)):
            j += 1
            file_name = f"{rank}_{suit}_{j}.jpg"

        # Simpan gambar di dalam subfolder
        folder_path = os.path.join("dataset", f"{rank}_{suit}")
        image_path = os.path.join(folder_path, file_name)

        # Buat subfolder jika belum ada
        os.makedirs(folder_path, exist_ok=True)

        cv2.imwrite(image_path, roi_resized)

        print(f"Image {j} captured: {file_name}")

    # Tombol 'o' untuk mengubah peringkat
    elif key == ord('o'):
        # Ganti indeks peringkat dan jenis kartu
        current_rank_index = (current_rank_index + 1) % len(ranks)
        if ranks[current_rank_index] == "2":
            current_suit_index = (current_suit_index + 1) % len(suits)

        j = 1

    # Hentikan program ketika tombol 'q' ditekan
    elif key == ord('q'):
        # Tambahkan satu ke jumlah percobaan setelah break
        break

# Bebaskan sumber daya dan tutup jendela
cap.release()
cv2.destroyAllWindows()
