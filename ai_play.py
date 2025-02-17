import game
import tensorflow as tf
import numpy as np
import time

# เช็คว่าใช้ GPU ได้หรือไม่
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✅ ใช้ GPU สำหรับ Play!")
else:
    print("⚠️ ไม่พบ GPU, ใช้ CPU แทน")

# โหลดโมเดลที่เทรนไว้
model = tf.keras.models.load_model("ai_pong-v1.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})


def play_ai():
    """ ให้ AI เล่นเกมโดยใช้โมเดลที่โหลดมา """
    while True:
        game.reset()  # รีเซ็ตเกมก่อนเริ่ม
        print("🚀 AI เริ่มเล่นเกม!")

        state, ballXSpeed, ballYSpeed = game.get_state()  # รับสถานะเริ่มต้น
        state = np.expand_dims(state, axis=-1).astype(np.float32) / 2  # เพิ่มมิติและ Normalize
        game_over = False
        
        while not game_over:
            # ให้ AI ทำนาย Action จากสถานะเกม
            q_values = model.predict(state.reshape(1, game.sceneSize, game.sceneSize, 1), verbose=0)  # ลด log
            action = np.argmax(q_values)

            print(action)

            # ควบคุม Paddle ตาม AI ทำนาย
            if action == 0:
                game.move_left()
            elif action == 2:
                game.move_right()

            # อัปเดตเกม 1 เฟรม
            hit, game_over = game.tick()

            # แสดงผลเกม
            #game.render()

            # รับสถานะใหม่
            state, ballXSpeed, ballYSpeed = game.get_state()
            state = np.expand_dims(state, axis=-1).astype(np.float32) / 2  # Normalize ค่าใหม่

            # เว้นระยะให้เกมแสดงผล
            time.sleep(0.2)

        print("💀 AI แพ้! รีสตาร์ทเกม...\n")

if __name__ == "__main__":
    play_ai()
