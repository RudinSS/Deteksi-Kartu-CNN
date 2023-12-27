import cv2
import numpy as np
import random
from keras.models import load_model

# Load model
model = load_model("DeteksiKartu.h5")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)  # Ganti angka 0 dengan indeks kamera jika memiliki lebih dari satu kamera
detected_player_frame = np.zeros((400, 400, 3), dtype=np.uint8)  
detected_bot_frame = np.zeros((400, 400, 3), dtype=np.uint8)  

# Tentukan ranks dan suits untuk setiap putaran
ranks = [str(i) for i in range(2, 11)] + ["jack", "queen", "king", "ace"]
suits = ["diamonds", "hearts", "spades", "clubs"]

# Ukuran kartu remi (lebar x tinggi)
card_width, card_height = 240, 360

# Menghitung koordinat untuk menempatkan kotak di bawah tengah frame
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
roi_x = (frame_width - card_width) // 2
roi_y = frame_height - card_height - 50

# Inisialisasi total poin
player_points = 0
bot_points = 0

# Dimulai dengan giliran player
player_turn = random.choice([True, False])
player_stand = False


# List untuk menyimpan kartu yang sudah terdeteksi
detected_cards = set()
last_detected_card = None

# Direktori tempat gambar kartu disimpan
card_images_dir = "PNG/"

def read_card_image(label):
    # Dapatkan nama file gambar kartu sesuai label
    image_path = f"{card_images_dir}{label}.png"
    # Baca gambar kartu
    card_image = cv2.imread(image_path)
    # Pastikan gambar terbaca dengan benar
    if card_image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return card_image


# Loop untuk mengambil gambar dari kamera
while True:
    # Ambil frame dari kamera
    key = cv2.waitKey(1)
    ret, frame = cap.read()
    roi = frame[roi_y:roi_y + card_height, roi_x:roi_x + card_width]
    # Konversi ROI ke ukuran yang lebih kecil
    roi_resized = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_AREA)
    roi_resized = roi_resized.astype("float32") / 255.0
    roi_resized = np.expand_dims(roi_resized, axis=0)
    
    # Prediksi kelas kartu menggunakan model
    pred = model.predict(roi_resized)

    # Loop untuk menghasilkan label
    LabelKelas = []
    for rank in ranks:
        for suit in suits:
            label = f"{rank}_{suit}"
            LabelKelas.append(label)

    # Tampilkan hasil prediksi
    label = LabelKelas[np.argmax(pred)]
    # Jika prediksi memiliki kepercayaan di atas ambang batas tertentu, tampilkan label jika tidak, tampilkan "Unknown"
    confidence_threshold = 0.5 
    if np.max(pred) > confidence_threshold:
        # Hitung poin kartu
        rank = label.split("_")[0]
        if rank in ["2", "3", "4", "5", "6", "7", "8", "9"]:
            points = int(rank)
        elif rank in ["10", "jack", "queen", "king"]:
            points = 10
        else:
            points = 11
            
        # Pemain memilih "stand" atau "hit"
        key = cv2.waitKey(1)
        if player_turn: 
            if key == ord('s'):  # Jika tombol 's' ditekan, pemain memilih "stand"
                player_stand = True
            elif key == ord('h'):  # Jika tombol 'h' ditekan, pemain memilih "hit"
                if label not in detected_cards:
                    player_points += points
                    frame_to_display = detected_player_frame
                    detected_cards.add(label)
                    # Tambahkan tulisan kartu ke frame detected_card_frame
                    card_image = read_card_image(label)
                    resized_card_image = cv2.resize(card_image, (75, 75))
                    detected_player_frame[len(detected_cards) * 20:len(detected_cards) * 20 + resized_card_image.shape[0], 10:10 + resized_card_image.shape[1]] = resized_card_image
                    player_turn = False
            
        if not player_turn:
            # Keputusan bot hit atau stand
            if bot_points < 17:
                # Bot memilih "hit" dan secara acak memilih kartu
                random_rank = random.choice(ranks)
                random_suit = random.choice(suits)
                label = f"{random_rank}_{random_suit}"

                # Jika kartu belum terdeteksi sebelumnya, tambahkan poin ke total bot
                if label not in detected_cards:
                    detected_cards.add(label)
                    # Tambahkan tulisan kartu ke frame detected_card_frame
                    card_image = read_card_image(label)
                    resized_card_image = cv2.resize(card_image, (75, 75))
                    detected_bot_frame[len(detected_cards) * 20:len(detected_cards) * 20 + resized_card_image.shape[0], 10:10 + resized_card_image.shape[1]] = resized_card_image

                    # Update total poin bot
                    rank = label.split("_")[0]
                    if rank in ["2", "3", "4", "5", "6", "7", "8", "9"]:
                        points = int(rank)
                    elif rank in ["10", "jack", "queen", "king"]:
                        points = 10
                    else:
                        points = 11
                    bot_points += points
                    # Ganti giliran ke pemain setelah bot "hit"
                    player_turn = True
            else:
                player_turn = True
        
        if player_stand:   
            # Keputusan bot hit atau stand
            player_turn = False
            if bot_points < 17:
                # Bot memilih "hit" dan secara acak memilih kartu
                random_rank = random.choice(ranks)
                random_suit = random.choice(suits)
                label = f"{random_rank}_{random_suit}"
                
                # Jika kartu belum terdeteksi sebelumnya, tambahkan poin ke total bot
                if label not in detected_cards:
                    detected_cards.add(label)
                    # Tambahkan tulisan kartu ke frame detected_card_frame
                    card_image = read_card_image(label)
                    resized_card_image = cv2.resize(card_image, (75, 75))
                    detected_bot_frame[len(detected_cards) * 20:len(detected_cards) * 20 + resized_card_image.shape[0], 10:10 + resized_card_image.shape[1]] = resized_card_image

                    # Update total poin bot
                    rank = label.split("_")[0]
                    if rank in ["2", "3", "4", "5", "6", "7", "8", "9"]:
                        points = int(rank)
                    elif rank in ["10", "jack", "queen", "king"]:
                        points = 10
                    else:
                        points = 11
                    bot_points += points
                    # Ganti giliran ke pemain setelah bot "hit"
                    player_turn = True
                else:
                    player_stand = False  # Atur player_stand menjadi False di sini    
            else:
                # Tentukan pemenang setelah loop selesai
                if player_points == bot_points or (bot_points > 21 and player_points > 21):
                        winner = "Draw"
                elif player_points > 21 or (bot_points <= 21 and bot_points > player_points):
                    winner = "Bot"
                elif bot_points > 21 or (player_points <= 21 and player_points > bot_points):
                    winner = "Player"
                else:
                    # Pilih pemenang berdasarkan total poin tertinggi
                    winner = "Player" if player_points > bot_points else "Bot"
                    # Tampilkan hasil permainan pada frame terakhir
                cv2.putText(frame, f"The winner is: {winner}", (220,180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)             
      
        # Update last_detected_card
        last_detected_card = label
        cv2.putText(detected_player_frame, "Player Card:", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(detected_bot_frame, "Bot Card:", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{label} ({points})", (roi_x + 5, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
   
    else: # jika tidak terdeteksi kartu
        if player_turn:
            if key == ord('s'):  # Jika tombol 's' ditekan, pemain memilih "stand"
                player_stand = True
        if player_stand:
            if bot_points >= 17:
              # Tentukan pemenang setelah loop selesai
                if player_points == bot_points or (bot_points > 21 and player_points > 21):
                        winner = "Draw"
                elif player_points > 21 or (bot_points <= 21 and bot_points > player_points):
                    winner = "Bot"
                elif bot_points > 21 or (player_points <= 21 and player_points > bot_points):
                    winner = "Player"
                else:
                    # Pilih pemenang berdasarkan total poin tertinggi
                    winner = "Player" if player_points > bot_points else "Bot"
                    # Tampilkan hasil permainan pada frame terakhir
                cv2.putText(frame, f"The winner is: {winner}", (220,180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)  
    # Gambar kotak pada frame
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + card_width, roi_y + card_height), (0, 0, 0), 5)
    
    # Tampilkan player & bot poin
    cv2.putText(frame, f"Player Points: {player_points}/21", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Bot Points: {bot_points}/21", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    #Tampilan Text Instruksi
    cv2.putText(frame, f"Hit press H and Stand press S", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Tampilkan frame 
    cv2.imshow("Detected Cards - Player", detected_player_frame)
    cv2.imshow("detection card", frame)
    cv2.imshow("Detected Cards - Bot", detected_bot_frame)
    
    
    # Hentikan program ketika tombol 'q' ditekan
    if key == ord('q'):
        break
    


# Tutup jendela
cap.release()
cv2.destroyAllWindows()
