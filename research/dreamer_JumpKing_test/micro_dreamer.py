import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
import pygame
import random
import time
import os

# ==========================================
# CONFIGURATION
# ==========================================
MODE = "train" 
# Penalize wasting time? (Makes it try to go faster)
TIME_PENALTY = 0.003

# ==========================================
# 1. THE ENVIRONMENT
# ==========================================
class MiniJumpKing(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self):
        self.window_size = 64
        self.action_space = spaces.Discrete(6) 
        self.observation_space = spaces.Box(0, 255, (3, 64, 64), dtype=np.uint8)
        
        self.world_height = 320 
        self.world_width = 64
        self.player_width = 6
        self.player_height = 6
        
        self.platforms = [pygame.Rect(0, self.world_height - 10, 64, 10)]
        current_y = self.world_height - 10
        while current_y > 40:
            gap = random.randint(22, 28) 
            current_y -= gap
            w = random.randint(24, 34)   
            x = random.randint(0, 64 - w)
            self.platforms.append(pygame.Rect(x, current_y, w, 4))
        self.platforms.append(pygame.Rect(0, 0, 64, 10)) 
        
        pygame.init()
        pygame.display.set_caption(f"MicroDreamer - {MODE.upper()}")
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.world_surface = pygame.Surface((self.world_width, self.world_height))
        
        self.reset()

    def reset(self):
        self.player_pos = np.array([32.0, self.world_height - 20.0])
        self.player_vel = np.array([0.0, 0.0])
        self.on_ground = False
        self.jump_cooldown = 0 
        return self._get_obs()

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); exit()

        if self.jump_cooldown > 0: self.jump_cooldown -= 1
        self.player_vel[1] += 0.25 

        if action == 1 or action == 4: self.player_vel[0] = -1.5
        elif action == 2 or action == 5: self.player_vel[0] = 1.5
            
        if self.on_ground and self.jump_cooldown == 0:
            if action in [3, 4, 5]: 
                self.player_vel[1] = -4.2 
                self.on_ground = False
                self.jump_cooldown = 8

        self.player_pos[0] += self.player_vel[0]
        self.player_vel[0] *= 0.8 
        if self.player_pos[0] < 0: self.player_pos[0] = 64
        if self.player_pos[0] > 64: self.player_pos[0] = 0

        self.player_pos[1] += self.player_vel[1]
        
        # Collision Logic
        if self.player_pos[1] < 0: self.player_pos[1] = 0; self.player_vel[1] = 0
        if self.player_pos[1] > self.world_height: self.player_pos[1] = self.world_height - 10; self.player_vel[1] = 0; self.on_ground = True

        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.player_width, self.player_height)
        
        if self.jump_cooldown == 0:
            self.on_ground = False 
            for plat in self.platforms:
                if player_rect.colliderect(plat):
                    if self.player_vel[1] > 0 and (self.player_pos[1] + self.player_height) < (plat.bottom):
                        self.player_pos[1] = plat.top - self.player_height
                        self.player_vel[1] = 0
                        self.on_ground = True

        # Base Reward: Height (0.0 to 1.0)
        reward = (self.world_height - self.player_pos[1]) / float(self.world_height)
        
        # Bonus: Reaching Top (Goal)
        if self.player_pos[1] < 10: 
            reward += 1.0 
            
        # Optimization: Time Penalty (Encourage Speed)
        reward -= TIME_PENALTY

        return self._get_obs(), reward, False, False, {}

    def _get_obs(self):
        self.world_surface.fill((0, 0, 0))
        for p in self.platforms: pygame.draw.rect(self.world_surface, (0, 255, 0), p)
        pygame.draw.rect(self.world_surface, (0, 0, 255), 
                         pygame.Rect(self.player_pos[0], self.player_pos[1], self.player_width, self.player_height))
        
        cam_y = self.player_pos[1] - (self.window_size // 2)
        if cam_y < 0: cam_y = 0
        if cam_y > (self.world_height - self.window_size): cam_y = self.world_height - self.window_size
        
        view = self.world_surface.subsurface(0, cam_y, 64, 64).copy()
        self.window.blit(view, (0,0))
        pygame.display.update()
        return np.transpose(np.array(pygame.surfarray.pixels3d(view)), (2, 0, 1))

# ==========================================
# 2. MODELS & UTILS
# ==========================================
class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 128), nn.ReLU()
        )
        self.rnn = nn.GRUCell(134, 256) 
        self.reward_head = nn.Linear(256, 1)
        self.decoder_linear = nn.Linear(256, 4096)
        self.decoder_net = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )
    def forward(self, obs, action, hidden):
        embed = self.encoder(obs / 255.0) 
        action_onehot = F.one_hot(action.long(), 6).float()
        rnn_input = torch.cat([embed, action_onehot], dim=1)
        next_hidden = self.rnn(rnn_input, hidden)
        pred_reward = self.reward_head(next_hidden)
        x = self.decoder_linear(next_hidden)
        pred_image = self.decoder_net(x)
        return pred_image, pred_reward, next_hidden

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 6))
        self.critic = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

def save_checkpoint(world_model, agent, filename):
    torch.save({'world_model': world_model.state_dict(), 'agent': agent.state_dict()}, filename)
    print(f">> SAVED: {filename}")

def load_checkpoint(world_model, agent, filename, device):
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=device)
        world_model.load_state_dict(checkpoint['world_model'])
        agent.load_state_dict(checkpoint['agent'])
        print(f">> LOADED: {filename}")
    else:
        print(f">> NOT FOUND: {filename} (Starting fresh)")

# ==========================================
# 3. MAIN LOOP (Optimized)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MicroDreamer started on {device} in {MODE.upper()} mode.")

    env = MiniJumpKing()
    world_model = WorldModel().to(device)
    agent = ActorCritic().to(device)
    
    # NEW LOAD STRATEGY:
    # Always try to load the BEST model first, so we train from the peak.
    if os.path.exists("micro_dreamer_best.pth"):
        print(">> FOUND BEST MODEL: Loading from peak performance!")
        load_checkpoint(world_model, agent, "micro_dreamer_best.pth", device)
    else:
        # Fallback to regular if best doesn't exist
        load_checkpoint(world_model, agent, "micro_dreamer.pth", device)

    opt_world = optim.Adam(world_model.parameters(), lr=1e-3)
    opt_agent = optim.Adam(agent.parameters(), lr=1e-4)
    
    buffer = []
    
    # TRACKING BEST PERFORMANCE
    # We set this low so the first jump will trigger a save
    max_reward_ever = -1.0 
    
    obs = env.reset()
    hidden = torch.zeros(1, 256).to(device) 

    step = 0
    try:
        while True:
            step += 1
            
            with torch.no_grad():
                logits = agent.actor(hidden)
                action = torch.distributions.Categorical(logits=logits).sample().item()
                if MODE == "train" and random.random() < 0.15: 
                    action = env.action_space.sample()

            next_obs, reward, _, _, _ = env.step(action)
            
            # Update Internal State
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                act_tensor = torch.tensor([action]).to(device)
                _, _, hidden = world_model(obs_tensor, act_tensor, hidden)

            # --- OPTIMIZATION: SAVE BEST MODEL ---
            # If this is the highest we've ever been (or close to top), SAVE IT immediately.
            if MODE == "train" and reward > max_reward_ever:
                max_reward_ever = reward
                print(f"!!! NEW RECORD: {reward:.4f} !!! Saving Best Model...")
                save_checkpoint(world_model, agent, "micro_dreamer_best.pth")

            if step % 100 == 0:
                print(f"Step {step} | Curr: {reward:.3f} | Best Ever: {max_reward_ever:.3f}")

            if MODE == "train":
                buffer.append((obs, action, reward, next_obs))
                if len(buffer) > 20000: buffer.pop(0)

                if len(buffer) > 100:
                    batch = random.sample(buffer, 32)
                    b_obs = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(device)
                    b_act = torch.tensor(np.array([x[1] for x in batch])).to(device)
                    b_rew = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32).unsqueeze(1).to(device)
                    b_hid = hidden.detach().repeat(32, 1)
                    
                    pred_img, pred_rew, _ = world_model(b_obs, b_act, b_hid)
                    loss_world = F.mse_loss(pred_img, b_obs / 255.0) + F.mse_loss(pred_rew, b_rew)
                    opt_world.zero_grad(); loss_world.backward(); opt_world.step()
                    
                    for p in world_model.parameters(): p.requires_grad = False
                    dream_hid = b_hid.detach(); dream_loss = 0
                    for _ in range(15): 
                        action_logits = agent.actor(dream_hid)
                        act_dream = torch.distributions.Categorical(logits=action_logits).sample()
                        _, _, dream_hid = world_model(b_obs, act_dream, dream_hid) 
                        value = agent.critic(dream_hid)
                        dream_loss -= value.mean()
                    opt_agent.zero_grad(); dream_loss.backward(); opt_agent.step()
                    for p in world_model.parameters(): p.requires_grad = True

                # Save Regular Checkpoint periodically
                if step % 500 == 0:
                    save_checkpoint(world_model, agent, "micro_dreamer.pth")

            obs = next_obs
            if MODE == "play": time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nStopping...")
        if MODE == "train":
            save_checkpoint(world_model, agent, "micro_dreamer.pth")
        pygame.quit()

if __name__ == "__main__":
    main()