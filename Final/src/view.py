import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def get_img_views(env, obs, sid, info):
    progress = info['progress']
    lap = int(info['lap'])
    score = lap + progress - 1.

    # Get the images
    img1 = env.env.env.force_render(render_mode='rgb_array_higher_birds_eye', width=540, height=540,
                                position=np.array([4.89, -9.30, -3.42]), fov=120)
    img2 = env.env.env.force_render(render_mode='rgb_array_birds_eye', width=270, height=270)
    img3 = env.env.env.force_render(render_mode='rgb_array_follow', width=128, height=128)
    obs = cv2.resize(obs, (128, 128))
    img4 = cv2.merge([obs,obs,obs])

    # Combine the images
    img = np.zeros((540, 810, 3), dtype=np.uint8)
    img[0:540, 0:540, :] = img1
    img[:270, 540:810, :] = img2
    img[270 + 10:270 + 128 + 10, 540 + 7:540 + 128 + 7, :] = img3
    img[270 + 10:270 + 128 + 10, 540 + 128 + 14:540 + 128 + 128 + 14, :] = img4

    # Draw the text
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('./res/Arial.ttf', 25)
    font_large = ImageFont.truetype('./res/Arial.ttf', 35)
    draw.text((5, 5), "Full Map", font=font, fill=(255, 87, 34))
    draw.text((550, 10), "Bird's Eye", font=font, fill=(255, 87, 34))
    draw.text((550, 280), "Follow", font=font, fill=(255, 87, 34))
    draw.text((688, 280), "Obs", font=font, fill=(255, 87, 34))
    draw.text((550, 408), f"Lap {lap}", font=font, fill=(255, 255, 255))
    draw.text((688, 408), f"Prog {progress:.3f}", font=font, fill=(255, 255, 255))
    draw.text((550, 450), f"Score {score:.3f}", font=font_large, fill=(255, 255, 255))
    draw.text((550, 500), f"ID {sid}", font=font_large, fill=(255, 255, 255))

    img = np.asarray(img)

    return img

def record_video(imgs, filename: str):
    height, width, layers = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filename, fourcc, 30, (width, height))
    for image in imgs:
        video.write(image)
    cv2.destroyAllWindows()
    video.release()

if __name__ =='__main__':
    from stable_baselines3 import PPO
    import sys,os
    from myEnv import FinalEnv
    model = PPO.load(sys.argv[1])
    test_env = FinalEnv(True, FinalEnv.austria_competition)
    
    obs, info = test_env.reset()

    done = False
    total_rew = 0
    local_step = 0
    
    imgs = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rew, term, trun, info = test_env.step(action)
        imgs.append(get_img_views(test_env, obs[-1], '123', info))

        done = term or trun

        total_rew += info['original_reward']
        local_step += 1
    
    record_video(imgs, f'video_{int(total_rew*100)}_{local_step}.mp4')
