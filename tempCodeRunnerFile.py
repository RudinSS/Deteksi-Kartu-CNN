
detected_player_frame = np.zeros((400, 400, 3), dtype=np.uint8)  
detected_bot_frame = np.zeros((400, 400, 3), dtype=np.uint8)  

# Tentukan ranks dan suits untuk setiap putaran
ranks = [str(i) for i in range(2, 11)] + ["jack", "queen", "king", "ace"]
suits = ["diamonds", "hearts", "spades", "clubs"]